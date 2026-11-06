"""Skill-declared MCP servers: manifest, binding gates, Connect tools and the lister report (ouroboros/skill_mcp.py)."""

from __future__ import annotations

import json
import pathlib
import queue
import sys
import types
import typing

import pytest

from ouroboros import config, mcp_client, mcp_catalog, skill_mcp
from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
from ouroboros.skill_loader import load_skill

DECL = ('[{"id": "chrome_browser", "title": "Chrome (shared tab)", "entry": "scripts/server.mjs", '
        '"task_session": true, "browser_policy": true}]')
OTHER = {"id": "github", "name": "GitHub", "enabled": True, "transport": "streamable_http",
         "url": "https://api.example/mcp", "auth_token": "sk-live-secret", "headers": {"X-Tenant": "acme"}}


def _manifest(servers: str = DECL, *, runtime: str = "node", kind: str = "script") -> str:
    return (f"---\nname: chrome_browser\ndescription: d\nversion: 0.3.0\ntype: {kind}\nruntime: {runtime}\n"
            f"scripts: [setup.mjs]\nmcp_servers: {servers}\n---\nbody\n")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """An installed skill directory, an isolated settings document, and a controllable readiness."""
    data = tmp_path / "data"
    skill_dir = data / "skills" / "external" / "chrome_browser"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_manifest(), encoding="utf-8")
    (skill_dir / "scripts" / "server.mjs").write_text("// server v1\n", encoding="utf-8")
    (skill_dir / "scripts" / "setup.mjs").write_text("// setup\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    state = types.SimpleNamespace(blockers=[], finds=0)

    def find(_root, name):
        state.finds += 1
        return load_skill(skill_dir, data) if name == "chrome_browser" and (skill_dir / "SKILL.md").exists() else None

    monkeypatch.setattr(skill_mcp, "_find", find)
    monkeypatch.setattr(skill_mcp, "_not_ready", lambda _root, _skill: list(state.blockers))
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary",
                        lambda runtime: ("/opt/runtime/bin/" + runtime, ""))
    state.data, state.skill_dir = data, skill_dir
    return state


def _write_settings(world, document):
    path = world.data / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _bound_row(**extra):
    return {"id": "chrome_browser", "name": "Chrome (shared tab)", "enabled": True, "transport": "stdio",
            "skill": "chrome_browser", "task_session": True, "browser_policy": True, **extra}


# --- manifest -------------------------------------------------------------------------------


def test_manifest_declares_servers_with_defaults_and_refuses_damage():
    manifest = parse_skill_manifest_text(_manifest())
    assert manifest.mcp_servers == [{"id": "chrome_browser", "title": "Chrome (shared tab)", "runtime": "node",
                                     "entry": "scripts/server.mjs", "task_session": True, "browser_policy": True}]
    assert manifest.validate() == [] and "mcp_servers" not in manifest.raw_extra
    minimal = parse_skill_manifest_text(_manifest('[{"id": "srv", "entry": "srv.mjs", "runtime": "node"}]'))
    assert minimal.mcp_servers[0] == {"id": "srv", "title": "srv", "runtime": "node", "entry": "srv.mjs",
                                      "task_session": False, "browser_policy": False}
    for bad, words in [('[{"id": "Chrome-Browser", "entry": "a.mjs"}]', "canonical"),
                       ('[{"id": "a", "entry": "../x.mjs"}]', "relative path"),
                       ('[{"id": "a", "entry": "a.mjs", "command": "node"}]', "unknown keys"),
                       ('[{"id": "a", "entry": "a.mjs", "browser_policy": true}]', "without task_session"),
                       ('[{"id": "a", "entry": "a.mjs"}, {"id": "a", "entry": "b.mjs"}]', "twice"),
                       ('{"id": "a"}', "must be a list")]:
        with pytest.raises(SkillManifestError, match=words):
            parse_skill_manifest_text(_manifest(bad))
    with pytest.raises(SkillManifestError, match="runtime"):
        parse_skill_manifest_text(_manifest('[{"id": "a", "entry": "a.mjs"}]', runtime=""))
    instruction = parse_skill_manifest_text(_manifest(kind="instruction").replace("scripts: [setup.mjs]\n", ""))
    assert any("mcp_servers" in warning for warning in instruction.validate())


# --- settings row and identity -----------------------------------------------------------------


def test_bound_row_parses_without_command_and_refuses_stored_launch_fields():
    errors: list = []
    cfg = mcp_client.normalize_server_config(_bound_row(), errors=errors)
    assert errors == [] and cfg.skill == "chrome_browser" and cfg.command == "" and cfg.args == []
    for extra in ({"command": "/usr/bin/node"}, {"args": ["x"]}, {"cwd": "/tmp"}, {"transport": "sse", "url": "http://x"}):
        assert mcp_client.normalize_server_config({**_bound_row(), **extra}, errors=errors) is None
    assert "takes command, args and cwd from its skill" in errors[0]


def test_unbound_identity_is_the_previous_formula_and_bound_identity_follows_the_revision():
    import dataclasses
    import hashlib

    plain = mcp_client.normalize_server_config({"id": "files", "transport": "stdio", "command": "npx", "args": ["-y"]})
    fields = dataclasses.asdict(plain)
    for key in ("enabled", "allowed_tools", "name", "configuration_warnings", "secret_values", "schemas_at_start",
                "task_session", "browser_policy", "skill", "skill_revision", "skill_error"):
        fields.pop(key)
    assert mcp_catalog.identity(plain) == hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    bound = mcp_client.normalize_server_config(_bound_row())
    one = dataclasses.replace(bound, command="/a/node", args=["/s/x"], cwd="/s", skill_revision="r1")
    moved = dataclasses.replace(one, command="/b/node", args=["/moved/x"], cwd="/moved", skill_error="")
    assert mcp_catalog.identity(one) == mcp_catalog.identity(moved)  # a relocated runtime keeps the catalog
    assert mcp_catalog.identity(one) != mcp_catalog.identity(dataclasses.replace(one, skill_revision="r2"))


def test_bind_derives_the_launch_and_refuses_every_not_ready_state(world):
    cfg = mcp_client.normalize_server_config(_bound_row())
    bound = skill_mcp.bind_config(cfg)
    assert bound.skill_error == "" and bound.command == "/opt/runtime/bin/node"
    assert bound.args == [str((world.skill_dir / "scripts" / "server.mjs").resolve())]
    assert bound.cwd == str(world.skill_dir.resolve()) and bound.skill_revision
    world.blockers = ["review_not_executable:review_pending"]
    assert "review_not_executable" in skill_mcp.bind_config(cfg).skill_error
    world.blockers = ["skill_disabled"]
    off = skill_mcp.bind_config(bound)
    assert off.command == "" and off.args == [] and off.skill_revision == "" and "skill_disabled" in off.skill_error
    world.blockers = []
    (world.skill_dir / "SKILL.md").write_text(_manifest('[{"id": "other", "entry": "scripts/server.mjs"}]'))
    assert "no longer declares server 'chrome_browser'" in skill_mcp.bind_config(cfg).skill_error
    (world.skill_dir / "SKILL.md").write_text(_manifest('[{"id": "chrome_browser", "entry": "node_modules/x.mjs"}]'))
    assert "cache or dependency folder" in skill_mcp.bind_config(cfg).skill_error
    (world.skill_dir / "SKILL.md").unlink()
    assert "not installed" in skill_mcp.bind_config(cfg).skill_error
    plain = mcp_client.normalize_server_config({"id": "files", "transport": "stdio", "command": "npx"})
    finds = world.finds
    assert skill_mcp.bind_config(plain) is plain and world.finds == finds  # unbound rows never reach a skill


# --- Connect tools -----------------------------------------------------------------------------


def _connect(world, *, confirm=False, digest=None):
    preview = skill_mcp.preview_connect(world.data, "chrome_browser")
    return preview, skill_mcp.connect(world.data, "chrome_browser", expected_content_hash=preview["content_hash"],
                                      settings_digest=preview["settings_digest"] if digest is None else digest,
                                      confirm_other_servers=confirm)


def test_connect_writes_exactly_the_bound_row_and_the_flag_preserving_everything_else(world):
    before = {"OUROBOROS_MODEL": "provider/model-x", "MCP_SERVERS": [OTHER], "MCP_ENABLED": True,
              "CUSTOM_SECRET_KEY": "sk-custom", "unknown_key": {"nested": [1, 2]},
              "OUROBOROS_STARTER_SKILLS_TURN_ON": False, "MCP_TOOL_TIMEOUT_SEC": "75"}  # saved spellings stay exact
    path = _write_settings(world, before)
    preview, result = _connect(world)
    assert preview["ready"] and preview["changed"] and not preview["other_servers"]
    assert preview["servers"][0]["change"] == "add" and preview["servers"][0]["row"] == _bound_row()
    assert preview["servers"][0]["executable"]["command"] == "/opt/runtime/bin/node"
    assert result["ok"] and result["saved"] and result["changed_keys"] == ["MCP_SERVERS"]
    after = json.loads(path.read_text())
    assert after["MCP_SERVERS"] == [OTHER, _bound_row()]  # the other row, its secret and header kept as written
    assert {k: v for k, v in after.items() if k != "MCP_SERVERS"} == {k: v for k, v in before.items() if k != "MCP_SERVERS"}
    stamp = path.stat().st_mtime_ns
    again_preview, again = _connect(world)
    assert again_preview["servers"][0]["change"] == "unchanged" and not again_preview["changed"]
    assert {k: v for k, v in again.items() if k != "servers"} == {
        "ok": True, "saved": False, "changed": False, "skill": "chrome_browser", "mcp_enabled": True, "other_servers": []}
    assert [s["change"] for s in again["servers"]] == ["unchanged"]
    assert path.stat().st_mtime_ns == stamp  # an identical connect writes nothing


def test_turning_mcp_on_discloses_and_requires_consent_for_other_servers(world):
    path = _write_settings(world, {"MCP_SERVERS": [OTHER, {**OTHER, "id": "off", "enabled": False}]})
    preview, refused = _connect(world)
    assert preview["requires_other_servers_confirmation"] and preview["other_servers"] == [{"id": "github", "name": "GitHub"}]
    assert refused["code"] == "other_servers_unconfirmed" and refused["saved"] is False
    assert "MCP_ENABLED" not in json.loads(path.read_text())
    _preview, done = _connect(world, confirm=True)
    after = json.loads(path.read_text())
    assert done["saved"] and after["MCP_ENABLED"] is True and after["MCP_SERVERS"][:2] == [OTHER, {**OTHER, "id": "off", "enabled": False}]


def test_connect_refuses_stale_previews_conflicts_and_not_ready_skills_without_writing(world):
    path = _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": []})
    preview = skill_mcp.preview_connect(world.data, "chrome_browser")
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [], "THEME": "dark"})  # changed after the preview
    stale = skill_mcp.connect(world.data, "chrome_browser", expected_content_hash=preview["content_hash"],
                              settings_digest=preview["settings_digest"])
    assert stale["code"] == "settings_precondition_failed" and json.loads(path.read_text())["MCP_SERVERS"] == []
    moved = skill_mcp.connect(world.data, "chrome_browser", expected_content_hash="0" * 64,
                              settings_digest=skill_mcp.preview_connect(world.data, "chrome_browser")["settings_digest"])
    assert moved["code"] == "skill_revision_changed"
    for row in ({"id": "chrome_browser", "transport": "stdio", "command": "node"},  # added by hand
                {**_bound_row(), "skill": "someone_else"}):
        _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [row]})
        preview, result = _connect(world)
        assert preview["conflict"] and preview["servers"][0]["change"] == "conflict" and preview["servers"][0]["row"] is None
        assert result["code"] == "mcp_server_conflict" and json.loads(path.read_text())["MCP_SERVERS"] == [row]
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [_bound_row(), _bound_row(name="twin")]})
    assert _connect(world)[1]["code"] == "mcp_server_conflict"
    assert skill_mcp.connect(world.data, "chrome_browser", expected_content_hash=preview["content_hash"],
                             settings_digest="")["code"] == "settings_digest_required"
    path.write_text("{not json")
    assert skill_mcp.preview_connect(world.data, "chrome_browser")["code"] == "settings_unreadable"
    world.blockers = ["review_not_executable:review_pending"]
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": []})
    preview, result = _connect(world)
    assert not preview["ready"] and preview["blockers"] and result["code"] == "skill_not_ready"
    assert json.loads(path.read_text())["MCP_SERVERS"] == []


def test_explicit_connect_re_enables_and_repairs_only_its_own_row(world):
    own = {**_bound_row(enabled=False), "allowed_tools": ["browser_snapshot"], "name": "My Chrome"}
    path = _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [own]})
    preview, result = _connect(world)
    assert preview["servers"][0]["change"] == "enable" and result["saved"]
    assert json.loads(path.read_text())["MCP_SERVERS"] == [{**own, "enabled": True}]  # owner choices kept
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [{**own, "command": "/old/node", "task_session": False}]})
    preview, _ = _connect(world)
    assert preview["servers"][0]["change"] == "repair"
    assert json.loads(path.read_text())["MCP_SERVERS"] == [{**own, "enabled": True}]


def test_owner_action_carries_the_same_connect_through_the_lifecycle_lane(world, monkeypatch):
    from ouroboros import skill_lifecycle_actions as actions
    from ouroboros.tools import skill_exec

    _write_settings(world, {"MCP_SERVERS": [OTHER]})
    ctx = types.SimpleNamespace(drive_root=world.data, repo_dir=world.data, task_id="", current_chat_id=0)
    monkeypatch.setattr(actions, "find_skill", lambda _drive, name, repo_path=None: load_skill(world.skill_dir, world.data))
    monkeypatch.setattr(actions, "canonical_data_root", lambda _ctx: world.data)
    refused = json.loads(skill_exec._handle_skill_owner_action(
        ctx, "chrome_browser", "connect_mcp", "", {}, preview=True).split("BLOCKED: ", 1)[1])
    assert refused["status_code"] == 403  # no task profile, no owner source: nothing is previewed or written
    preview = actions.preview_connect_mcp(ctx, "chrome_browser", _owner_actor="owner_ui")
    result = actions.run_skill_action(ctx, "chrome_browser", "connect_mcp", expected_content_hash=preview["content_hash"],
                                      settings_digest=preview["settings_digest"], confirm_other_servers=True,
                                      _owner_actor="owner_ui")
    assert result["ok"] and result["saved"]
    events = [json.loads(line) for line in (world.data / "logs" / "events.jsonl").read_text().splitlines()]
    audit = next(e for e in events if e.get("action") == "skill_connect_mcp")
    assert audit["other_servers"] == ["github"] and audit["confirm_other_servers"] is True
    assert audit["mcp_servers"] == [{"id": "chrome_browser", "change": "add"}] and audit["saved"] is True


# --- runtime gates: discovery, dispatch, update -----------------------------------------------


@pytest.fixture()
def manager(world, monkeypatch):
    retired: list = []
    monkeypatch.setattr("ouroboros.mcp_task_session.retire_changed", lambda configs: retired.append(list(configs)))
    mgr = mcp_client.MCPManager()
    listed: list = []

    async def list_tools(cfg, _timeout):
        listed.append(cfg)
        return [{"name": "browser_snapshot", "description": "snap", "inputSchema": {"type": "object"}}]

    async def call_tool(cfg, name, _args, _timeout):
        listed.append(("call", cfg, name))
        return mcp_client.ToolResult(status="ok", code="OK", text="called")

    mgr._async_list_tools, mgr._async_call_tool = list_tools, call_tool
    mgr.listed_calls, mgr.retired = listed, retired
    return mgr


def _settings(*rows, enabled=True):
    return {"MCP_ENABLED": enabled, "MCP_SERVERS": list(rows)}


def test_not_ready_skill_refuses_only_its_server_and_starts_nothing(world, manager):
    world.blockers = ["review_stale"]
    plain = {"id": "files", "transport": "stdio", "command": "npx", "enabled": True}
    manager.reconfigure(_settings(_bound_row(task_session=False, browser_policy=False), plain))
    refused = manager.refresh_server("chrome_browser")
    assert refused["not_started"] and refused["code"] == "MCP_SKILL_UNAVAILABLE" and "review_stale" in refused["error"]
    assert manager.refresh_server("files")["ok"] and [c.id for c in manager.listed_calls] == ["files"]
    result = manager._call_tool_result("mcp_chrome_browser__browser_snapshot", {})
    assert result.code == "MCP_UNAVAILABLE" and "MCP_SKILL_UNAVAILABLE" in result.text and "review_stale" in result.text
    assert manager._call_tool_result("mcp_files__browser_snapshot", {}).text.endswith("called")
    assert [getattr(c, "id", None) for c in manager.listed_calls] == ["files", None]  # only the unbound call ran
    status = {row["id"]: row for row in manager.status_payload()["servers"]}
    assert status["chrome_browser"]["skill"] == "chrome_browser" and "review_stale" in status["chrome_browser"]["skill_error"]
    assert status["files"]["skill_error"] == ""
    test = manager.test_server(_bound_row())
    assert test["code"] == "MCP_SKILL_UNAVAILABLE" and len(manager.listed_calls) == 2


def test_the_accepted_declaration_decides_the_call_lifetime_not_the_saved_row(world, manager):
    (world.skill_dir / "SKILL.md").write_text(_manifest('[{"id": "chrome_browser", "entry": "scripts/server.mjs", '
                                                        '"task_session": true}]'))
    stale_row = _bound_row(browser_policy=False)
    manager.reconfigure(_settings(stale_row))
    assert manager.status_payload()["servers"][0]["browser_policy"] is False
    (world.skill_dir / "SKILL.md").write_text(_manifest())  # an accepted update now asks for the browser guard
    retired = len(manager.retired)
    manager.rebind()
    status = manager.status_payload()["servers"][0]
    assert status["task_session"] is True and status["browser_policy"] is True and len(manager.retired) > retired
    assert skill_mcp.bind_config(mcp_client.normalize_server_config(stale_row)).browser_policy is True


def test_collateral_disclosure_reads_enabled_as_the_loader_does(world):
    rows = [{**OTHER, "id": "absent", "enabled": None}, {**OTHER, "id": "text_on", "enabled": "true"},
            {**OTHER, "id": "text_off", "enabled": "no"}]
    for row in rows:
        row.pop("enabled") if row["enabled"] is None else None
    _write_settings(world, _settings(*rows, enabled=False))
    loaded = {cfg.id: cfg.enabled for cfg in mcp_client.parse_servers(rows)}
    disclosed = [item["id"] for item in skill_mcp.preview_connect(world.data, "chrome_browser")["other_servers"]]
    assert disclosed == [server_id for server_id, on in loaded.items() if on] == ["text_on"]


def test_accepted_update_relists_and_retires_sessions_while_a_disabled_row_stays_off(world, manager):
    (world.skill_dir / "SKILL.md").write_text(_manifest('[{"id": "chrome_browser", "entry": "scripts/server.mjs"}]'))
    row = _bound_row(task_session=False, browser_policy=False)  # a per-call server: calls run in this test
    manager.reconfigure(_settings(row))
    assert manager.refresh_server("chrome_browser")["ok"]
    first = manager.listed_calls[-1]
    assert first.command == "/opt/runtime/bin/node" and first.skill_revision
    assert manager._call_tool_result("mcp_chrome_browser__browser_snapshot", {}).text.endswith("called")
    (world.skill_dir / "scripts" / "server.mjs").write_text("// server v2 (accepted)\n")
    events: queue.Queue = queue.Queue()
    stale = manager._call_tool_result("mcp_chrome_browser__browser_snapshot", {},
                                      types.SimpleNamespace(event_queue=events))
    assert stale.code == "MCP_CATALOG_UNAVAILABLE"  # the old listing is not served for the new bytes
    assert events.get_nowait() == {"type": "mcp_configuration_changed", "reason": "mcp_dispatch", "skill": ""}
    assert manager.retired, "the old revision's task sessions are retired"
    assert manager.refresh_server("chrome_browser")["ok"]
    assert manager.listed_calls[-1].skill_revision != first.skill_revision
    assert manager._call_tool_result("mcp_chrome_browser__browser_snapshot", {}).text.endswith("called")
    manager.reconfigure(_settings({**row, "enabled": False}))
    off = manager._call_tool_result("mcp_chrome_browser__browser_snapshot", {})
    assert "disabled" in off.text and "MCP_SKILL_UNAVAILABLE" not in off.text


def test_lister_answers_a_worker_report_without_restart_refresh_or_incidental_reads(world, manager, monkeypatch):
    lister, worker = manager, mcp_client.MCPManager()
    lister.become_lister()
    monkeypatch.setattr(worker, "_lister_pid", -1)
    calls: list = []
    def refresh(*, reason):
        calls.append(("list", reason))
        lister.refresh_all()

    monkeypatch.setattr(lister, "refresh_all_background", refresh)
    _write_settings(world, _settings())
    lister.configuration_changed(config.load_settings(), reason="startup")
    calls.clear()
    _write_settings(world, _settings(_bound_row()))  # a worker's Connect wrote this
    lister.reconfigure(config.load_settings())  # an incidental status read in the server
    monkeypatch.setattr(mcp_client, "get_manager", lambda: worker)
    events: queue.Queue = queue.Queue()
    skill_mcp.lifecycle_settled("chrome_browser", event_queue=events, reason="skill_connect_mcp")
    report = events.get_nowait()
    assert report == {"type": "mcp_configuration_changed", "reason": "skill_connect_mcp", "skill": "chrome_browser"}
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    from supervisor.events import dispatch_event

    dispatch_event(report, types.SimpleNamespace(DRIVE_ROOT=world.data, append_jsonl=lambda *_a, **_k: None))
    assert calls == [("list", "skill_connect_mcp")] and "chrome_browser" in lister.server_ids()
    calls.clear()
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")  # nothing moved: nothing is listed
    assert calls == []
    (world.skill_dir / "scripts" / "server.mjs").write_text("// accepted v3\n")
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")
    assert calls == [("list", "skill_review")]


@pytest.mark.parametrize("gate", ["ready", "mcp_disabled", "server_disabled", "review_pending", "skill_disabled"])
def test_accepted_update_is_discovered_after_passive_skills_get(world, manager, monkeypatch, gate):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway import extensions

    manager.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr(manager, "refresh_all_background", lambda *, reason: manager.refresh_all())
    monkeypatch.setattr(extensions, "_request_drive_root", lambda _request: world.data)
    monkeypatch.setattr(extensions, "discover_skills", lambda *_a, **_kw: [load_skill(world.skill_dir, world.data)])
    settings = _settings(_bound_row(enabled=gate != "server_disabled"), enabled=gate != "mcp_disabled")
    path = _write_settings(world, settings)
    manager.configuration_changed(config.load_settings(), reason="startup")
    before = len(manager.listed_calls)
    (world.skill_dir / "scripts" / "server.mjs").write_text("// updated payload\n")
    world.blockers = [gate] if gate in {"review_pending", "skill_disabled"} else []
    revision = load_skill(world.skill_dir, world.data).content_hash

    def disk_snapshot():
        return {p.relative_to(world.data): (p.stat().st_mtime_ns, p.read_bytes())
                for p in world.data.rglob("*") if p.is_file()}

    unchanged = disk_snapshot()
    with TestClient(Starlette(routes=[Route("/api/extensions", extensions.api_extensions_index)])) as client:
        response = client.get("/api/extensions")
    assert response.status_code == 200
    card = response.json()["skills"][0]["mcp_servers"][0]
    assert not card["listed"] and card["tool_count"] == 0
    assert disk_snapshot() == unchanged, "passive GET must not write settings, catalogs or skill state"
    assert len(manager.listed_calls) == before, "passive GET must not start discovery"
    assert manager.listed(config.load_settings()), "the settings fingerprint did not change"

    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")
    if gate == "ready":
        assert len(manager.listed_calls) == before + 1
        assert manager.listed_calls[-1].skill_revision == revision
        assert manager.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status == "callable"
        assert manager.retired[-1][0].skill_revision == revision
    else:
        assert len(manager.listed_calls) == before, "disabled or unaccepted payloads must start nothing"
        assert manager.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status != "callable"
    assert (path.stat().st_mtime_ns, path.read_bytes()) == unchanged[path.relative_to(world.data)]
    settled = len(manager.listed_calls)
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")
    assert len(manager.listed_calls) == settled, "settled evidence must not cause another listing"


def test_worker_reports_discovery_after_an_earlier_rebind_consumed_the_transition(world, manager, monkeypatch):
    from supervisor.events import dispatch_event

    lister, worker = manager, mcp_client.MCPManager()
    lister.become_lister()
    monkeypatch.setattr(lister, "refresh_all_background", lambda *, reason: lister.refresh_all())
    _write_settings(world, _settings(_bound_row()))
    settings = config.load_settings()
    lister.configuration_changed(settings, reason="startup")
    worker.reconfigure(settings)
    assert worker.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status == "callable"
    (world.skill_dir / "scripts" / "server.mjs").write_text("// accepted update\n")
    revision = load_skill(world.skill_dir, world.data).content_hash
    worker.rebind()  # no event queue at the first observation
    assert worker.status_payload()["servers"][0]["listed_at"] is None
    assert len(lister.listed_calls) == 1

    events: queue.Queue = queue.Queue()
    result = worker._call_tool_result("mcp_chrome_browser__browser_snapshot", {},
                                      types.SimpleNamespace(event_queue=events))
    assert result.code == "MCP_CATALOG_UNAVAILABLE" and len(lister.listed_calls) == 1
    assert not events.empty(), "an earlier reader must not consume the worker's owed discovery notification"
    report = events.get_nowait()
    assert report == {"type": "mcp_configuration_changed", "reason": "mcp_dispatch", "skill": ""}
    monkeypatch.setattr(mcp_client, "get_manager", lambda: lister)
    dispatch_event(report, types.SimpleNamespace(DRIVE_ROOT=world.data, append_jsonl=lambda *_a, **_kw: None))
    assert len(lister.listed_calls) == 2 and lister.listed_calls[-1].skill_revision == revision
    worker.rebind(event_queue=events)  # adopt the publication even though the config is already current
    assert worker.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status == "callable"
    assert events.empty(), "the current catalog settles the worker's discovery notification"


@pytest.mark.parametrize("outcome", ["empty", "failed"])
def test_skill_lifecycle_does_not_repeat_a_current_catalog_attempt(world, manager, monkeypatch, outcome):
    async def list_tools(cfg, _timeout):
        manager.listed_calls.append(cfg)
        if outcome == "failed":
            raise RuntimeError("server unavailable")
        return []

    manager._async_list_tools = list_tools
    manager.become_lister()
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr(manager, "refresh_all_background", lambda *, reason: manager.refresh_all())
    _write_settings(world, _settings(_bound_row()))
    manager.configuration_changed(config.load_settings(), reason="startup")
    status = manager.status_payload()["servers"][0]
    assert status["tool_count"] == 0 and status["latest_attempt"]["ok"] is (outcome == "empty")
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")
    assert len(manager.listed_calls) == 1


def test_import_never_rewrites_a_bound_row():
    from ouroboros.mcp_import import preview_import

    document = json.dumps({"mcpServers": {"chrome_browser": {"command": "node", "args": ["x"]}}})
    entry = preview_import(document, [{"id": "chrome_browser", "transport": "stdio", "skill": "chrome_browser"}])["entries"][0]
    assert entry["action"] == "" and entry["patch"] == {} and "skill 'chrome_browser'" in entry["problems"][0]


def test_connect_contract_twin_and_producers_agree(world):
    from ouroboros.gateway import skill_mcp_contracts as contracts
    from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
    from ouroboros.gateway.schema import validate_ingress
    from tests.test_gateway_parity import _js_typedef_fields

    twin = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "skill_mcp_types.js").read_text()
    for name in contracts.__all__:
        assert _js_typedef_fields(twin, name) == set(typing.get_type_hints(getattr(contracts, name))), name
    _write_settings(world, _settings())
    preview = skill_mcp.preview_connect(world.data, "chrome_browser")
    assert set(preview) <= set(typing.get_type_hints(contracts.SkillMcpConnectPreviewResponse))
    assert set(preview["servers"][0]) == set(typing.get_type_hints(contracts.SkillMcpConnectServer))
    result = skill_mcp.connect(world.data, "chrome_browser", expected_content_hash=preview["content_hash"],
                               settings_digest=preview["settings_digest"])
    assert set(result) <= set(typing.get_type_hints(contracts.SkillMcpConnectResponse))
    assert {"GET /api/skills/{skill}/mcp-connect", "POST /api/skills/{skill}/mcp-connect"} <= set(HTTP_ENDPOINTS)
    assert validate_ingress({"expected_content_hash": "h", "settings_digest": "d"}, contracts.SkillMcpConnectRequest) == []
    assert validate_ingress({"settings_digest": "d"}, contracts.SkillMcpConnectRequest)
    api_types = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js").read_text()
    assert "skill_mcp_types.js" in api_types.splitlines()[0]
    assert sys.modules["ouroboros.gateway.skill_mcp_contracts"] is contracts


# --- the agent's own path: skill_owner_action with a retained answered quiz -----------------------------


@pytest.fixture()
def agent_world(tmp_path, monkeypatch):
    """A real installed, accepted, enabled skill and a real task ToolContext; nothing in skill_mcp is stubbed."""
    from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state
    from ouroboros.tools.registry import ToolContext

    drive, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    skill_dir = drive / "skills" / "external" / "chrome_browser"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_manifest('[{"id": "chrome_browser", "title": "Chrome (shared tab)", '
                                                  '"entry": "scripts/server.mjs", "task_session": true, '
                                                  '"browser_policy": true}]'), encoding="utf-8")
    (skill_dir / "scripts" / "server.mjs").write_text("console.log('server');\n", encoding="utf-8")
    (skill_dir / "scripts" / "setup.mjs").write_text("console.log('setup');\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    revision = load_skill(skill_dir, drive).content_hash
    save_review_state(drive, "chrome_browser", SkillReviewState(status="clean", content_hash=revision))
    save_enabled(drive, "chrome_browser", True, actor="owner_ui")
    (drive / "settings.json").write_text(json.dumps({"MCP_SERVERS": [OTHER], "OUROBOROS_STARTER_SKILLS_TURN_ON": False}))
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id, ctx.current_chat_id = "connect-task", 42
    return types.SimpleNamespace(ctx=ctx, drive=drive, revision=revision)


def _agent_call(world, **kwargs):
    from ouroboros.tools.skill_exec import _handle_skill_owner_action

    rendered = _handle_skill_owner_action(world.ctx, "chrome_browser", "connect_mcp", world.revision, **kwargs)
    return json.loads(rendered.split("SKILL_ACTION_BLOCKED: ", 1)[-1]), rendered.startswith("⚠️")


def test_agent_connect_carries_an_answered_quiz_through_real_owner_source_validation(agent_world):
    from ouroboros.owner_quiz import record_answered, record_asked

    source = {"kind": "quiz", "task_id": "connect-task", "quiz_id": "connect-chrome"}
    record_asked(agent_world.drive, "connect-task", quiz_id="connect-chrome", options=["Connect", "Not now"],
                 question="Connect Chrome (shared tab)? Turning MCP on also starts GitHub.")
    unanswered, blocked = _agent_call(agent_world, owner_source=source, preview=True)
    assert blocked and unanswered["status_code"] == 403  # an asked but unanswered quiz is not owner intent
    record_answered(agent_world.drive, "connect-task", quiz_id="connect-chrome", option_index=0, request_id="answer-1")
    preview, blocked = _agent_call(agent_world, owner_source=source, preview=True)
    assert not blocked and preview["ready"] and preview["other_servers"] == [{"id": "github", "name": "GitHub"}]
    assert preview["servers"][0]["executable"]["args"][0].endswith("scripts/server.mjs")
    path = agent_world.drive / "settings.json"
    assert json.loads(path.read_text()) == {"MCP_SERVERS": [OTHER], "OUROBOROS_STARTER_SKILLS_TURN_ON": False}
    foreign = {**source, "task_id": "another-task"}
    refused, blocked = _agent_call(agent_world, owner_source=foreign, settings_digest=preview["settings_digest"],
                                   confirm_other_servers=True)
    assert blocked and refused["status_code"] == 403 and "MCP_ENABLED" not in json.loads(path.read_text())
    unconfirmed, blocked = _agent_call(agent_world, owner_source=source, settings_digest=preview["settings_digest"])
    assert blocked and unconfirmed["code"] == "other_servers_unconfirmed"
    done, blocked = _agent_call(agent_world, owner_source=source, settings_digest=preview["settings_digest"],
                                confirm_other_servers=True)
    assert not blocked and done["ok"] and done["saved"], done
    assert json.loads(path.read_text()) == {"MCP_SERVERS": [OTHER, _bound_row()], "MCP_ENABLED": True,
                                            "OUROBOROS_STARTER_SKILLS_TURN_ON": False}
    events = [json.loads(line) for line in (agent_world.drive / "logs" / "events.jsonl").read_text().splitlines()]
    audit = [row for row in events if row.get("action") == "skill_connect_mcp"][-1]
    assert audit["actor"] == "agent_tool" and audit["task_id"] == "connect-task" and audit["saved"] is True
    assert audit["source_ref"]["kind"] == "quiz" and audit["source_ref"]["quiz_id"] == "connect-chrome"
    assert audit["other_servers"] == ["github"] and audit["confirm_other_servers"] is True
