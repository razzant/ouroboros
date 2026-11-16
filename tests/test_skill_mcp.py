"""Skill-declared MCP servers: manifest, binding gates, Connect tools and the lister report (ouroboros/skill_mcp.py)."""

from __future__ import annotations

import dataclasses
import json
import os
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
    state = types.SimpleNamespace(blockers=[], finds=0, readiness=skill_mcp._not_ready)

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


def test_preflight_checks_each_declared_server_entry_as_its_start_does(tmp_path, monkeypatch):
    """skill_preflight (review, acceptance and owner attestation) checks every ``mcp_servers[].entry`` with the start's
    own rule (``skill_mcp.entry_problem``): an absent file, a folder, a symlink out of the skill, a dependency folder or
    a symlink into one fails; a reviewed file passes."""
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.skill_preflight import _handle_skill_preflight

    drive, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    skill_dir = drive / "skills" / "external" / "entries"
    (skill_dir / "scripts" / "folder.mjs").mkdir(parents=True)
    (skill_dir / "node_modules").mkdir()
    (skill_dir / ".ouroboros_env" / "node").mkdir(parents=True)
    for path in ("scripts/server.mjs", "scripts/setup.mjs", "node_modules/dep.mjs", ".ouroboros_env/node/env.mjs"):
        (skill_dir / path).write_text("// js\n", encoding="utf-8")
    (tmp_path / "outside.mjs").write_text("// outside the skill\n", encoding="utf-8")
    (skill_dir / "scripts" / "escape.mjs").symlink_to(tmp_path / "outside.mjs")
    (skill_dir / "scripts" / "cached.mjs").symlink_to(skill_dir / ".ouroboros_env" / "node" / "env.mjs")
    entries = {"good": "scripts/server.mjs", "absent": "scripts/missing.mjs", "folder": "scripts/folder.mjs",
               "escape": "scripts/escape.mjs", "dependency": "node_modules/dep.mjs", "cached": "scripts/cached.mjs"}
    servers = json.dumps([{"id": key, "entry": entry} for key, entry in entries.items()])
    (skill_dir / "SKILL.md").write_text(_manifest(servers).replace("name: chrome_browser", "name: entries"),
                                        encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    result = json.loads(_handle_skill_preflight(ToolContext(repo_dir=repo, drive_root=drive), skill="entries"))
    rows = [row for row in result["manifest"] if row["item"] == "manifest_mcp_entry_exists"]
    verdicts = {key: next(row["ok"] for row in rows if f"{key!r}" in row["detail"] or row["detail"] == entry)
                for key, entry in entries.items()}
    assert verdicts == {"good": True, "absent": False, "folder": False, "escape": False, "dependency": False,
                        "cached": False}, rows
    assert result["ok"] is False
    for key in ("dependency", "cached"):
        assert "cache or dependency folder" in next(row["detail"] for row in rows if f"{key!r}" in row["detail"])


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
                "task_session", "browser_policy", "skill", "skill_revision", "skill_settings", "skill_error"):
        fields.pop(key)
    assert mcp_catalog.identity(plain) == hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    bound = mcp_client.normalize_server_config(_bound_row())
    one = dataclasses.replace(bound, command="/a/node", args=["/s/x"], cwd="/s", skill_revision="r1")
    moved = dataclasses.replace(one, command="/b/node", args=["/moved/x"], cwd="/moved", skill_error="")
    assert mcp_catalog.identity(one) == mcp_catalog.identity(moved)  # a relocated runtime keeps the catalog
    assert mcp_catalog.identity(one) != mcp_catalog.identity(dataclasses.replace(one, skill_revision="r2"))
    assert mcp_catalog.identity(one) != mcp_catalog.identity(dataclasses.replace(one, skill_settings="other digest"))


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


def test_a_deployment_policy_refusal_of_the_write_is_typed_and_writes_nothing(world, monkeypatch):
    """The writer's own refusal of a managed choice (``fields_on_saved_document``) reaches the caller as a typed 403."""
    from ouroboros import deployment_policy

    path = _write_settings(world, {"MCP_SERVERS": [OTHER]})
    before = path.read_bytes()
    managed = "MCP_ENABLED is managed by this deployment's policy file; nothing was saved."
    monkeypatch.setattr(deployment_policy, "choice_refusal", lambda key, _value: managed if key == "MCP_ENABLED" else "")
    preview, result = _connect(world, confirm=True)
    assert preview["ready"] and preview["changed"]
    assert {key: result[key] for key in ("ok", "saved", "code", "status_code", "error")} == {
        "ok": False, "saved": False, "code": "managed_by_deployment", "status_code": 403, "error": managed}
    assert path.read_bytes() == before


def test_a_managed_work_domain_is_read_passively_and_refuses_only_what_it_cannot_start(world, monkeypatch):
    """Managed, the skill folder outside the domain's mappings: preview, Connect and the binding copy and start
    nothing (the preview names the work copy the domain will start); the real start owner (``work_start``) is the
    one that copies. Only what the domain cannot start is refused: a domain naming no staging_root, an extension's
    package, a skill changed since its binding. Unmanaged, or a folder inside the mappings, nothing changes."""
    from ouroboros import work_domain, work_skill_payloads

    path = _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": []})
    monkeypatch.setattr(skill_mcp, "_not_ready", world.readiness)  # the real gate: the domain decides before readiness
    monkeypatch.setattr(work_domain, "managed", lambda: True)
    monkeypatch.setattr(work_domain, "is_work", lambda _path: False)
    staging = world.data.parent / "work" / "staging"  # service-side path, before the execution domain's mapping
    monkeypatch.setattr(work_skill_payloads, "_home", lambda name, digest: staging / "skills" / name / digest)
    copies = []
    monkeypatch.setattr(work_skill_payloads, "launch", lambda *args: copies.append(args) or (["node"], "/srv", {}))
    monkeypatch.setattr("ouroboros.skill_readiness.skill_readiness_for_execution",
                        lambda *_a, **_k: types.SimpleNamespace(blockers=[]))
    preview, result = _connect(world)
    home = staging / "skills" / "chrome_browser" / preview["content_hash"]
    assert preview["ready"] and preview["servers"][0]["executable"] == {
        "command": "node", "args": [str(home / "scripts" / "server.mjs")], "cwd": str(home)}
    assert result["saved"] and json.loads(path.read_text())["MCP_SERVERS"] == [_bound_row()]
    bound = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    assert bound.skill_error == "" and bound.command == "node" and copies == []  # passive; the container's runtime
    argv, cwd, env = skill_mcp.work_start(bound, {"WITNESS": "w"})
    assert copies == [(world.data, copies[0][1], bound.skill_revision, [bound.command, *bound.args], copies[0][4],
                       "node")] and (argv, cwd, env) == (["node"], "/srv", {})  # the start is the one that copies
    assert copies[0][4]["WITNESS"] == "w" and copies[0][4]["OUROBOROS_SKILL_NAME"] == "chrome_browser"  # its env
    with pytest.raises(work_domain.WorkDomainRefused, match="changed since it was bound"):
        skill_mcp.work_start(dataclasses.replace(bound, skill_revision="0" * 64), {})
    assert len(copies) == 1
    extension = types.SimpleNamespace(skill_dir=world.skill_dir, name="x", content_hash="h",
                                      manifest=types.SimpleNamespace(is_extension=lambda: True))
    assert "an extension's package stays on the service" in skill_mcp._work_copy(extension)[1]
    with monkeypatch.context() as inside:  # even a folder inside the mappings: an extension's code is no work process
        inside.setattr(work_domain, "is_work", lambda _path: True)
        assert "an extension's package stays on the service" in skill_mcp._work_copy(extension)[1]

    def no_staging(_name, _digest):
        raise work_domain.WorkDomainRefused("This deployment's execution domain names no staging_root")

    monkeypatch.setattr(work_skill_payloads, "_home", no_staging)
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": []})
    preview, result = _connect(world)
    assert not preview["ready"] and preview["blockers"] == ["This deployment's execution domain names no staging_root"]
    assert preview["next_action"] == "" and result["code"] == "skill_not_ready"
    refused = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    assert refused.command == "" and refused.skill_error.endswith("this deployment decides where it runs.")
    monkeypatch.setattr(work_domain, "is_work", lambda _path: True)  # a skills checkout inside the mappings
    assert skill_mcp._work_copy(load_skill(world.skill_dir, world.data)) == (None, "")
    monkeypatch.setattr(work_domain, "managed", lambda: False)
    argv, cwd, env = skill_mcp.work_start(bound, {"A": "1"})
    assert (argv, cwd, env["A"]) == ([bound.command, *bound.args], bound.cwd, "1") and len(copies) == 1
    assert env["OUROBOROS_SKILL_NAME"] == "chrome_browser"  # unmanaged: the binding's start, with its skill's env


def test_a_managed_work_copy_needs_the_containers_runtime_not_the_services(world, monkeypatch):
    """Managed, the work container runs every skill server with its own interpreter, staged (a work copy,
    ``work_skill_payloads.launch``) or mapped (a skills folder inside the mappings), so a service with no usable Node
    still previews, connects, binds and starts it there, never asked. Unmanaged, the service names the interpreter:
    its runtime check stays and refuses."""
    from ouroboros import work_domain, work_skill_payloads

    path = _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": []})
    probed = []
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary",
                        lambda runtime: probed.append(runtime) or (None, "bundled:absent; path:missing"))
    monkeypatch.setattr(work_domain, "managed", lambda: True)
    monkeypatch.setattr(work_domain, "is_work", lambda _path: False)
    home = world.data.parent / "work-copy"
    monkeypatch.setattr(work_skill_payloads, "_home", lambda name, digest: str(home))
    copies = []
    monkeypatch.setattr(work_skill_payloads, "materialize_skill", lambda *args: copies.append(args) or home)
    preview, result = _connect(world)
    assert preview["blockers"] == [] and preview["ready"] and result["saved"]
    assert preview["servers"][0]["executable"] == {"command": "node", "args": [str(home / "scripts/server.mjs")],
                                                   "cwd": str(home)}
    assert json.loads(path.read_text())["MCP_SERVERS"] == [_bound_row()]
    bound = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    assert bound.skill_error == "" and bound.command == "node" and copies == [] and probed == []  # passive, no probe
    argv, cwd, _env = skill_mcp.work_start(bound, {})  # the real launch: the copy, named in the container
    assert argv == ["node", str(home / "scripts/server.mjs")] and cwd == str(home) and len(copies) == 1
    monkeypatch.setattr(work_domain, "is_work", lambda _path: True)  # mapped: still the container's runtime
    inside = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    entry = str(world.skill_dir.resolve() / "scripts/server.mjs")
    assert inside.skill_error == "" and (inside.command, inside.args) == ("node", [entry]) and probed == []
    argv, _cwd, _env = skill_mcp.work_start(inside, {})  # started where it is mapped: no work copy
    assert argv == ["node", entry] and len(copies) == 1 and probed == []
    monkeypatch.setattr(work_domain, "managed", lambda: False)  # unmanaged: the service names the interpreter
    unmanaged = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    assert unmanaged.command == "" and "no usable node runtime was found (bundled:absent; path:missing)" in (
        unmanaged.skill_error)
    assert probed == ["node"] and len(copies) == 1  # only the unmanaged binding asked the service


_WITNESSED = ("WITNESS_VALUE", "WITNESS_SECRET", "UNDECLARED_SECRET", "ROW_VALUE", "OWNER_PICK", "OUROBOROS_SKILL_NAME",
              "OUROBOROS_SKILL_STATE_DIR", "NODE_PATH", "OUROBOROS_BROWSER_GUARD")


def test_a_real_start_gets_its_skill_scripts_environment_and_nothing_undeclared(tmp_path, monkeypatch):
    """Every real stdio start (``_transport_factory``: listing, Test, calls, task sessions) gives a skill-bound
    server what ``skill_exec`` gives its scripts: granted declared settings, its name and state folder, its isolated
    packages; never a setting it did not declare and was not granted. The row's own env and settings references (the
    owner's explicit choices, not grants), then the host's entries overlay it, the skill's identity stays, granted and
    referenced secrets mask its text, a grant revoked since binding starts nothing, and an unbound server starts
    exactly as before."""
    import subprocess

    from mcp.client.stdio import get_default_environment

    from ouroboros import mcp_browser_guard, node_runtime, work_domain, workspace_executor
    from ouroboros.mcp_schema import _redact_error_text
    from ouroboros.process_scope import installation_scope_entries
    from ouroboros.skill_loader import (
        SkillReviewState,
        save_enabled,
        save_review_state,
        save_skill_grants,
        skill_state_dir_path,
    )
    from ouroboros.tools.skill_exec import _resolve_runtime_binary

    node = _resolve_runtime_binary("node")[0]
    if not node:
        pytest.skip("no usable Node runtime on this host")
    drive = tmp_path / "data"
    skill_dir = drive / "skills" / "external" / "env_witness"
    (skill_dir / ".ouroboros_env" / "node" / "node_modules").mkdir(parents=True)
    (skill_dir / "server.cjs").write_text(f"console.log(JSON.stringify(Object.fromEntries({json.dumps(list(_WITNESSED))}"
                                          ".map((k) => [k, process.env[k] ?? null]))));\n")
    (skill_dir / "skill.json").write_text(json.dumps({
        "name": "env_witness", "type": "script", "description": "env witness", "version": "1.0", "runtime": "node",
        "scripts": ["server.cjs"], "env_from_settings": ["WITNESS_VALUE", "WITNESS_SECRET"],
        "mcp_servers": [{"id": "env_witness", "entry": "server.cjs"}]}))
    settings_path = drive / "settings.json"
    settings_path.write_text(json.dumps({"WITNESS_VALUE": "fixture-value", "WITNESS_SECRET": "witness-secret-0123",
                                         "UNDECLARED_SECRET": "undeclared-secret-4567",
                                         "OWNER_SECRET": "owner-secret-7890", "MCP_SERVERS": []}))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(work_domain, "managed", lambda: False)
    monkeypatch.setattr("ouroboros.tools.skill_exec._resolve_runtime_binary", lambda _runtime: (node, ""))
    loaded = load_skill(skill_dir, drive)
    save_review_state(drive, loaded.name, SkillReviewState(status="clean", content_hash=loaded.content_hash))
    save_skill_grants(drive, loaded.name, ["WITNESS_VALUE", "WITNESS_SECRET"], content_hash=loaded.content_hash,
                      requested_keys=["WITNESS_VALUE", "WITNESS_SECRET"])
    save_enabled(drive, loaded.name, True, actor="owner_ui")
    preview = skill_mcp.preview_connect(drive, loaded.name)
    assert preview.get("ready"), preview
    assert skill_mcp.connect(drive, loaded.name, expected_content_hash=loaded.content_hash,
                             settings_digest=preview["settings_digest"], confirm_other_servers=True)["saved"]
    saved = json.loads(settings_path.read_text())
    saved["MCP_SERVERS"][0]["env"] = {"ROW_VALUE": "row"}  # the owner's literal variables on the bound row
    saved["MCP_SERVERS"][0]["env_from_settings"] = {"OWNER_PICK": "OWNER_SECRET"}  # the owner's choice, not a grant
    plain = {"id": "plain", "transport": "stdio", "command": "/bin/echo", "env": {"ROW_VALUE": "row"}, "enabled": True}
    cfg, unbound = (skill_mcp.bind_config(item) for item in mcp_client.parse_servers(
        [*saved["MCP_SERVERS"], plain], settings=saved))
    assert cfg.skill_error == "" and {"fixture-value", "witness-secret-0123"} <= set(cfg.secret_values)
    assert "undeclared-secret-4567" not in cfg.secret_values and skill_mcp.bind_config(cfg) == cfg  # stable rebind
    monkeypatch.setattr(mcp_client, "_stdio_with_diagnostics", lambda params, *_: params)
    host = {mcp_browser_guard.ADDRESS_ENV: "http://127.0.0.1:9/guard", "OUROBOROS_SKILL_STATE_DIR": "/elsewhere"}
    params = mcp_client._transport_factory(cfg, host)
    seen = json.loads(subprocess.check_output([params.command, *params.args], cwd=params.cwd, text=True,
                                              env={**get_default_environment(), **params.env}))
    assert seen == {
        "WITNESS_VALUE": "fixture-value", "WITNESS_SECRET": "witness-secret-0123", "UNDECLARED_SECRET": None,
        "ROW_VALUE": "row", "OWNER_PICK": "owner-secret-7890", "OUROBOROS_SKILL_NAME": "env_witness",
        "OUROBOROS_SKILL_STATE_DIR": str(skill_state_dir_path(drive, "env_witness")),
        "NODE_PATH": str(skill_dir / ".ouroboros_env" / "node" / "node_modules"),
        mcp_browser_guard.ADDRESS_ENV: "http://127.0.0.1:9/guard"}
    # An unbound local server: as before, no skill, plus Playwright's driver Node (the pin's prepared one, marked as
    # selected; an explicit operator value is kept as given, unmarked). Nothing else is added.
    driver, plain = "/managed/node-24.22.3", {**installation_scope_entries(), "ROW_VALUE": "row", **host}
    with monkeypatch.context() as pinned:
        pinned.setattr("ouroboros.claudexor_runtime.get_runtime_manager",
                       lambda: types.SimpleNamespace(reviewed_node=lambda prepared=False: driver))
        pinned.delenv(node_runtime.PLAYWRIGHT_NODE_ENV, raising=False)
        pinned.delenv(node_runtime.SELECTED_DRIVER_NODE_ENV, raising=False)
        assert mcp_client._transport_factory(unbound, host).env == {
            **plain, node_runtime.PLAYWRIGHT_NODE_ENV: driver, node_runtime.SELECTED_DRIVER_NODE_ENV: driver}
        pinned.setenv(node_runtime.PLAYWRIGHT_NODE_ENV, "/operator/node")
        assert mcp_client._transport_factory(unbound, host).env == {
            **plain, node_runtime.PLAYWRIGHT_NODE_ENV: "/operator/node"}
    masked = _redact_error_text("echo witness-secret-0123 fixture-value owner-secret-7890 undeclared-secret-4567", cfg)
    assert not {"witness-secret-0123", "fixture-value", "owner-secret-7890"} & set(masked.split())
    assert "undeclared-secret-4567" in masked  # neither granted nor referenced: never sent, so never masked
    # The owner's reference is part of the identity as before: its changed value is a new identity.
    other = {**saved, "OWNER_SECRET": "owner-secret-rotated"}
    rebound = skill_mcp.bind_config(mcp_client.parse_servers(saved["MCP_SERVERS"], settings=other)[0])
    assert mcp_catalog.identity(rebound) != mcp_catalog.identity(cfg) and "owner-secret-rotated" in rebound.secret_values
    # The row's literal variables outrank a granted value; the skill's identity outranks both; Windows ignores case.
    _argv, _cwd, ordered = skill_mcp.work_start(cfg, {"WITNESS_VALUE": "owner-row", "OUROBOROS_SKILL_NAME": "spoof"})
    assert (ordered["WITNESS_VALUE"], ordered["OUROBOROS_SKILL_NAME"]) == ("owner-row", "env_witness")
    monkeypatch.setattr(workspace_executor, "IS_WINDOWS", True)
    _argv, _cwd, windows = skill_mcp.work_start(cfg, {"Path": "/row/bin"})
    assert windows["Path"] == "/row/bin" and "PATH" not in windows  # one entry, never a Path/PATH pair
    monkeypatch.setattr(workspace_executor, "IS_WINDOWS", False)
    # A granted value is part of what a start receives: its change is a new identity (a task session restarts on it),
    # an unchanged one keeps the identity, and the identity holds a digest, never the value.
    same = skill_mcp.bind_config(cfg)
    assert mcp_catalog.identity(same) == mcp_catalog.identity(cfg) and "witness-secret-0123" not in json.dumps(
        {key: value for key, value in dataclasses.asdict(cfg).items() if key != "secret_values"})
    settings_path.write_text(json.dumps({**json.loads(settings_path.read_text()), "WITNESS_SECRET": "rotated-0456"}))
    rotated = skill_mcp.bind_config(cfg)
    from ouroboros.mcp_task_session import session_identity

    assert mcp_catalog.identity(rotated) != mcp_catalog.identity(cfg)
    assert session_identity(rotated) != session_identity(cfg)
    assert "rotated-0456" in rotated.secret_values  # masked from its first start on
    # A start racing the owner's toggle-off or an edit after binding starts nothing, here as under a managed domain.
    save_enabled(drive, loaded.name, False, actor="owner_ui")
    with pytest.raises(work_domain.WorkDomainRefused, match="MCP_SKILL_UNAVAILABLE.*skill_disabled") as refused:
        mcp_client._transport_factory(rotated, host)
    assert refused.value.mcp_unsent_joined is True  # refused before any transport: a proven no-send
    save_enabled(drive, loaded.name, True, actor="owner_ui")
    reviewed = (skill_dir / "server.cjs").read_text()
    (skill_dir / "server.cjs").write_text(reviewed + "// edited after the binding\n")
    with pytest.raises(work_domain.WorkDomainRefused, match="changed since it was bound"):
        mcp_client._transport_factory(rotated, host)
    (skill_dir / "server.cjs").write_text(reviewed)
    monkeypatch.setattr("ouroboros.skill_loader.grant_status_for_skill", lambda *_a: {
        "missing_keys": ["WITNESS_SECRET"], "granted_keys": ["WITNESS_VALUE"]})  # revoked since the binding
    with pytest.raises(work_domain.WorkDomainRefused, match="MCP_SKILL_UNAVAILABLE") as refused:
        mcp_client._transport_factory(rotated, host)
    assert refused.value.mcp_unsent_joined is True


@pytest.mark.parametrize("blocker", ["skill_disabled", "review_stale", "deps_missing", "skill_conflict:calendar"])
@pytest.mark.parametrize("managed", [False, True])
def test_a_start_rechecks_what_its_binding_checked(world, monkeypatch, blocker, managed):
    """Every real start re-reads the skill's readiness as ``bind_config`` did: a toggle-off, a stale review, missing
    packages or a conflict arising after the binding starts nothing, unmanaged or managed (no work copy is made)."""
    from ouroboros import work_domain, work_skill_payloads

    bound = skill_mcp.bind_config(mcp_client.normalize_server_config(_bound_row()))
    assert bound.skill_error == ""
    copies = []
    monkeypatch.setattr(work_domain, "managed", lambda: managed)
    monkeypatch.setattr(work_domain, "is_work", lambda _path: True)  # managed: a skills checkout inside the mappings
    monkeypatch.setattr(work_skill_payloads, "launch", lambda *args: copies.append(args) or (["node"], "/srv", {}))
    world.blockers = [blocker]  # arose after the binding
    with pytest.raises(work_domain.WorkDomainRefused, match=f"MCP_SKILL_UNAVAILABLE.*{blocker}"):
        skill_mcp.work_start(bound, {})
    assert copies == []
    world.blockers = []
    assert skill_mcp.work_start(bound, {})[0] == (["node"] if managed else [bound.command, *bound.args])


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
    # The owner's own env and settings references are the owner's choices: a repair and a Connect both keep them.
    choices = {"env": {"DEBUG": "1"}, "env_from_settings": {"TOKEN": "OWNER_TOKEN"}}
    _write_settings(world, {"MCP_ENABLED": True, "MCP_SERVERS": [{**own, **choices, "command": "/old/node"}]})
    preview, _ = _connect(world)
    assert preview["servers"][0]["change"] == "repair"
    assert json.loads(path.read_text())["MCP_SERVERS"] == [{**own, **choices, "enabled": True}]
    preview, _ = _connect(world)
    assert preview["servers"][0]["change"] == "unchanged"
    assert json.loads(path.read_text())["MCP_SERVERS"] == [{**own, **choices, "enabled": True}]


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
    yield mgr
    assert mgr.discovery.wait(10), "the fixture owns MCP discovery until it settles"


def _settings(*rows, enabled=True):
    return {"MCP_ENABLED": enabled, "MCP_SERVERS": list(rows)}


@pytest.mark.parametrize("state", ["partial", "failed_after_list", "skill_error", "mcp_off", "server_off", "unbound", "conflict"])
def test_card_facts_separate_catalog_notes_failures_and_off(world, manager, state):
    settings = _settings(_bound_row())
    _write_settings(world, settings)
    manager.become_lister()
    manager.reconfigure(settings)

    async def partial(*_args):
        return [{"name": "browser_snapshot", "inputSchema": {"type": "object"}}, {"_pagination_truncated": True}]

    manager._async_list_tools = partial
    manager.refresh_all()
    assert manager.resolve_tool_name("mcp_chrome_browser__browser_snapshot").status == "callable"
    if state == "failed_after_list":
        async def fail(*_args):
            raise RuntimeError("Fixture latest listing failed")
        manager._async_list_tools = fail
        manager.refresh_all()
    elif state == "skill_error":
        world.blockers = ["skill_disabled"]
        manager.rebind()
    elif state == "mcp_off":
        settings["MCP_ENABLED"] = False
    elif state == "server_off":
        settings["MCP_SERVERS"][0]["enabled"] = False
    elif state == "unbound":
        settings["MCP_SERVERS"] = []
    elif state == "conflict":
        settings["MCP_SERVERS"][0]["skill"] = "other"
    status = {row["id"]: row for row in manager.status_payload()["servers"]}
    fact, = skill_mcp.card_facts(load_skill(world.skill_dir, world.data), settings, status)
    if state == "partial":
        assert fact["error"] == ""
        assert fact["notes"] == status["chrome_browser"]["last_error"] and fact["notes"]
        assert fact["listed"] and fact["tool_count"] == 1
    elif state in {"failed_after_list", "skill_error"}:
        assert ("Fixture latest listing failed" if state == "failed_after_list" else "skill_disabled") in fact["error"]
        if state == "failed_after_list":
            assert fact["listed"] and fact["tool_count"] == 1
    assert fact["bound"] is (state not in {"unbound", "conflict"})
    assert fact["configured"] is (state not in {"mcp_off", "server_off", "unbound", "conflict"})
    if state in {"unbound", "conflict"}:
        assert not fact["error"] and not fact["notes"] and not fact["listed"]
    assert fact["conflict"] is (state == "conflict")


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


@pytest.mark.parametrize("initial_state", ["accepted_v1", "review_pending", "skill_disabled",
                                           "source_install", "peer_conflict"])
def test_registry_worker_adopts_published_skill_catalog_before_admission(world, manager, monkeypatch, initial_state):
    """A long-lived worker must admit the lister's new revision without a direct-call rebind."""
    from unittest.mock import Mock
    from ouroboros.tools.registry import ToolRegistry

    manifest = _manifest('[{"id": "chrome_browser", "entry": "scripts/server.mjs"}]')
    (world.skill_dir / "SKILL.md").write_text(manifest)
    blocker = {"source_install": "not installed", "peer_conflict": "skill_conflict:calendar"}.get(initial_state, initial_state)
    world.blockers = [] if initial_state in {"accepted_v1", "source_install"} else [blocker]
    if initial_state == "source_install":
        (world.skill_dir / "SKILL.md").unlink()
    elif initial_state == "peer_conflict":
        from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state

        peer = world.skill_dir.parent / "calendar"
        peer.mkdir()
        (peer / "SKILL.md").write_text("---\nname: calendar\ndescription: peer\ntype: instruction\n"
                                      "conflicts: [chrome_browser]\n---\n")
        save_enabled(world.data, "calendar", True)
        save_enabled(world.data, "chrome_browser", True)
        save_review_state(world.data, "chrome_browser", SkillReviewState(
            status="clean", content_hash=load_skill(world.skill_dir, world.data).content_hash))
        monkeypatch.setattr(skill_mcp, "_not_ready", world.readiness)
        assert world.readiness(world.data, load_skill(world.skill_dir, world.data)) == [blocker]
    path = _write_settings(world, _settings(_bound_row(task_session=False, browser_policy=False)))
    saved = path.stat().st_mtime_ns, path.read_bytes()
    settings = config.load_settings()
    lister, worker = manager, mcp_client.MCPManager()
    lister.become_lister()
    monkeypatch.setattr(lister, "refresh_all_background", lambda *, reason: lister.refresh_all())
    lister.configuration_changed(settings, reason="startup")
    worker.reconfigure(settings, settings_mtime_ns=saved[0])
    monkeypatch.setattr(mcp_client, "get_manager", lambda: worker)
    worker._async_list_tools = Mock(side_effect=AssertionError("a worker must never discover tools"))
    dispatched = []

    async def call(cfg, name, _args, _timeout):
        dispatched.append((cfg, name))
        return mcp_client.ToolResult(status="ok", code="OK", text="called")

    worker._async_call_tool = call
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    registry = ToolRegistry(repo_dir=world.data, drive_root=world.data)
    registry._ctx.event_queue = queue.Queue()
    name = "mcp_chrome_browser__browser_snapshot"
    initial = registry.execute_result(name, {})
    if initial_state == "accepted_v1":
        assert initial.status == "ok" and len(dispatched) == 1
    else:  # resolved before Safety as the skill's own fact, not as incomplete discovery
        assert initial.code == "MCP_UNAVAILABLE" and "MCP_SKILL_UNAVAILABLE" in initial.text and not dispatched
        assert blocker in initial.text and blocker in worker._servers["chrome_browser"].config.skill_error
    old = worker._servers["chrome_browser"].config

    if initial_state == "source_install":
        (world.skill_dir / "SKILL.md").write_text(manifest)
    elif initial_state == "peer_conflict":
        save_enabled(world.data, "calendar", False)
    else:
        (world.skill_dir / "scripts" / "server.mjs").write_text("// server v2 (accepted)\n")
    # Disabling a conflicting peer changes readiness, not this skill's payload or Settings row.
    world.blockers = []
    revision = load_skill(world.skill_dir, world.data).content_hash

    async def list_updated(cfg, _timeout):
        lister.listed_calls.append(cfg)
        return [{"name": raw, "description": raw, "inputSchema": {"type": "object"}}
                for raw in ("browser_snapshot", "browser_new")]

    lister._async_list_tools = list_updated
    with monkeypatch.context() as server:
        server.setattr(mcp_client, "get_manager", lambda: lister)
        target, reason = {"source_install": ("ouroboros/chrome_browser", "skill_install"),
                          "peer_conflict": ("calendar", "skill_disable")}.get(initial_state, ("chrome_browser", "skill_review"))
        skill_mcp.lifecycle_settled(target, reason=reason)
        assert lister.discovery.wait(10)
    assert lister.listed_calls, "lifecycle completion must list a newly ready binding even when its target differs"
    assert lister.listed_calls[-1].skill_revision == revision != old.skill_revision
    assert worker._servers["chrome_browser"].config is old, "the worker has not observed the lifecycle event"
    catalog = mcp_catalog.catalog_path("chrome_browser")
    published = catalog.stat().st_mtime_ns, catalog.read_bytes()
    assert json.loads(published[1])["identity"] == mcp_catalog.identity(lister.listed_calls[-1])
    listed, retired, called = len(lister.listed_calls), len(manager.retired), len(dispatched)

    # No worker.rebind(), schema read, settings write or restart between publication and real admission.
    results = [registry.execute_result(tool, {}) for tool in (name, name, "mcp_chrome_browser__browser_new")]
    assert [result.code for result in results] == ["OK"] * 3, [result.text for result in results]
    assert [raw for _, raw in dispatched[called:]] == ["browser_snapshot", "browser_snapshot", "browser_new"]
    assert all(cfg.skill_revision == revision and not cfg.skill_error for cfg, _ in dispatched[called:])
    assert len(manager.retired) > retired and manager.retired[-1][0].skill_revision == revision
    worker._async_list_tools.assert_not_called()
    assert not worker.is_lister and len(lister.listed_calls) == listed and registry._ctx.event_queue.empty()
    assert (path.stat().st_mtime_ns, path.read_bytes()) == saved
    assert (catalog.stat().st_mtime_ns, catalog.read_bytes()) == published


def test_registry_refuses_an_unready_skill_before_safety_and_keeps_every_other_fact(world, manager, monkeypatch):
    """The real ToolRegistry seam: an unready skill's tool is ``MCP_SKILL_UNAVAILABLE`` before Safety, while a
    disabled row stays disabled, a manual server never listed stays a catalog gap (no implicit discovery) and a
    ready skill's listed tool is called."""
    from ouroboros.tools.registry import ToolRegistry

    checked = []
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: checked.append(_a) or (True, ""))
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    (world.skill_dir / "SKILL.md").write_text(_manifest('[{"id": "chrome_browser", "entry": "scripts/server.mjs"}]'))
    plain = {"id": "files", "transport": "stdio", "command": "npx", "enabled": True}
    registry = ToolRegistry(repo_dir=world.data, drive_root=world.data)
    registry._ctx.event_queue = queue.Queue()
    skill_tool, manual_tool = "mcp_chrome_browser__browser_snapshot", "mcp_files__read"

    def outcome(name):
        result = registry.execute_result(name, {})
        return result.code, result.text

    world.blockers = ["skill_disabled"]
    _write_settings(world, _settings(_bound_row(task_session=False, browser_policy=False), plain))
    code, text = outcome(skill_tool)
    assert code == "MCP_UNAVAILABLE" and "MCP_SKILL_UNAVAILABLE" in text and "skill_disabled" in text
    code, text = outcome(manual_tool)
    assert code == "MCP_CATALOG_UNAVAILABLE" and "files" in text
    assert checked == [] and manager.listed_calls == []  # no Safety, no discovery for either refusal
    _write_settings(world, _settings(_bound_row(enabled=False, task_session=False, browser_policy=False), plain))
    code, text = outcome(skill_tool)
    assert code == "MCP_UNAVAILABLE" and "is disabled in Settings" in text and "MCP_SKILL_UNAVAILABLE" not in text
    world.blockers = []
    _write_settings(world, _settings(_bound_row(task_session=False, browser_policy=False), plain))
    mcp_client.ensure_configured_from_settings()
    assert manager.refresh_server("chrome_browser")["ok"]  # the explicit listing the ready binding is owed
    code, text = outcome(skill_tool)
    assert code == "OK" and len(checked) == 1, text


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
    assert lister.discovery.wait(10)
    assert calls == [("list", "skill_connect_mcp")] and "chrome_browser" in lister.server_ids()
    calls.clear()
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")  # nothing moved: nothing is listed
    assert manager.discovery.wait(10)
    assert calls == []
    (world.skill_dir / "scripts" / "server.mjs").write_text("// accepted v3\n")
    skill_mcp.lifecycle_settled("chrome_browser", reason="skill_review")
    assert manager.discovery.wait(10)
    assert lister.listed_calls[-1].skill_revision == load_skill(world.skill_dir, world.data).content_hash


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
    assert manager.discovery.wait(10)
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
    assert manager.discovery.wait(10)
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
    assert lister.discovery.wait(10)
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
    assert manager.discovery.wait(10)
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


def _published_200(payload, method="POST", status="200"):
    """The names of the route's published bodies for ``status`` (``endpoint_index`` via the gateway ABI) ``payload``
    matches."""
    from ouroboros.gateway.schema import contract_document, contract_typed_dict, validate_ingress

    entry = next(item for item in contract_document()["http"]
                 if (item["method"], item["path"]) == (method, "/api/skills/{skill}/mcp-connect"))
    return [name for name in entry["responses"][status]["schemas"]
            if not validate_ingress(payload, contract_typed_dict(name))]


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("segment", ["%20", "%20%09"])
def test_a_blank_skill_segment_answers_a_published_refusal(world, method, segment):
    """``/api/skills/%20/mcp-connect``: a typed refusal in the route's published 400 body, before any lane or job."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import extensions

    app = Starlette(routes=[Route("/api/skills/{skill}/mcp-connect", extensions.api_skill_mcp_connect,
                                  methods=["GET", "POST"])])
    app.state.drive_root = app.state.repo_dir = world.data
    with TestClient(app) as client:
        body = {"expected_content_hash": "h", "settings_digest": "d"} if method == "POST" else None
        response = client.request(method, f"/api/skills/{segment}/mcp-connect", headers={"Prefer": "respond-async"},
                                  json=body)
    payload = response.json()
    assert response.status_code == 400 and payload["code"] == "missing_skill" and payload["ok"] is False
    assert _published_200(payload, method, "400"), payload


@pytest.mark.parametrize("queued", [True, False])
def test_an_async_connect_answers_a_published_shape_and_pending_never_claims_saved(world, manager, monkeypatch, queued):
    """``Prefer: respond-async`` through the real route and lifecycle queue: a confirm past the answer bound is HTTP
    200 ``SkillMcpConnectPending``, with no ``saved`` and no settled outcome (here its held writer has not run yet),
    and its job then saves; one that settles in time is its ``SkillMcpConnectResponse``. Both are published 200
    bodies of the route."""
    import asyncio
    import threading

    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import skill_lifecycle_actions, skill_lifecycle_queue
    from ouroboros.gateway import extensions

    path = _write_settings(world, _settings(enabled=False))
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)
    monkeypatch.setattr(skill_lifecycle_actions, "find_skill", lambda *_a, **_kw: load_skill(world.skill_dir, world.data))
    monkeypatch.setattr(skill_lifecycle_actions, "canonical_data_root", lambda _ctx: world.data)
    monkeypatch.setattr(skill_lifecycle_queue, "JOB_ANSWER_WAIT_SEC", 0.05 if queued else 30)
    monkeypatch.setattr(extensions, "_broadcast_extension_lifecycle", lambda *_a, **_kw: None)
    release, completed, real_connect = threading.Event(), threading.Event(), skill_mcp.connect

    def held_connect(*args, **kwargs):  # the writer waits until the request has been answered
        assert release.wait(10), "the test never released the writer"
        try:
            return real_connect(*args, **kwargs)
        finally:
            completed.set()

    monkeypatch.setattr(skill_mcp, "connect", held_connect)
    if not queued:
        release.set()
    preview = skill_mcp.preview_connect(world.data, "chrome_browser")
    app = Starlette(routes=[Route("/api/skills/{skill}/mcp-connect", extensions.api_skill_mcp_connect, methods=["POST"])])
    app.state.drive_root = app.state.repo_dir = world.data
    with TestClient(app) as client:
        try:
            response = client.post("/api/skills/chrome_browser/mcp-connect", headers={"Prefer": "respond-async"},
                                   json={"expected_content_hash": preview["content_hash"],
                                         "settings_digest": preview["settings_digest"], "confirm_other_servers": True})
            payload = response.json()
            written_while_pending = json.loads(path.read_text())["MCP_SERVERS"]
        finally:
            release.set()
            assert completed.wait(10)

            async def drain():
                await asyncio.gather(*tuple(skill_lifecycle_queue._BACKGROUND))

            client.portal.call(drain)
    assert response.status_code == 200
    if queued:
        assert payload["pending"] is True and payload["job_id"] and "saved" not in payload and payload["ok"] is True
        assert _published_200(payload) == ["SkillMcpConnectPending"] and written_while_pending == []
    else:
        assert payload["saved"] is True and "pending" not in payload
        assert _published_200(payload) == ["SkillMcpConnectResponse"]
    assert json.loads(path.read_text())["MCP_SERVERS"] == [_bound_row()]  # the job saved it either way


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


def test_agent_connect_waits_for_an_unprepared_pin_even_with_an_answered_quiz(agent_world, monkeypatch):
    """The preparation-absent control: a node skill waits for the pin's Node (owner, 14 Nov 2026); a real answered
    quiz carries owner intent, never a runtime, so the preview names the unprepared pin and the confirmed Connect
    writes nothing."""
    from ouroboros import node_runtime, platform_layer
    from ouroboros.claudexor_runtime import get_runtime_manager
    from ouroboros.owner_quiz import record_answered, record_asked

    # Empty DATA alone does not exclude an exact packaged Node or a carried prepared generation.
    monkeypatch.setattr(platform_layer, "bundled_resource_bases", lambda: [])
    monkeypatch.delenv(node_runtime.CARRIED_NODE_ENV, raising=False)
    manager = get_runtime_manager()  # read from the pin itself, not from the policy under test
    if not manager.pin_error and not manager.planned_node(manager.pin):
        pytest.skip("the pin has no Node archive for this host, so its skills keep the PATH rollback")
    source = {"kind": "quiz", "task_id": "connect-task", "quiz_id": "connect-chrome"}
    record_asked(agent_world.drive, "connect-task", quiz_id="connect-chrome", options=["Connect", "Not now"],
                 question="Connect Chrome (shared tab)? Turning MCP on also starts GitHub.")
    record_answered(agent_world.drive, "connect-task", quiz_id="connect-chrome", option_index=0, request_id="answer-1")
    preview, blocked = _agent_call(agent_world, owner_source=source, preview=True)
    assert not blocked and not preview["ready"] and preview["servers"][0]["executable"] == {}
    assert len(preview["blockers"]) == 1 and "chrome_browser" in preview["blockers"][0]
    assert ":not_prepared" in preview["blockers"][0] and "no usable node runtime" in preview["blockers"][0]
    path = agent_world.drive / "settings.json"
    refused, blocked = _agent_call(agent_world, owner_source=source, settings_digest=preview["settings_digest"],
                                   confirm_other_servers=True)
    assert blocked and refused["code"] == "skill_not_ready" and ":not_prepared" in refused["error"]
    assert json.loads(path.read_text()) == {"MCP_SERVERS": [OTHER], "OUROBOROS_STARTER_SKILLS_TURN_ON": False}


def test_agent_connect_carries_an_answered_quiz_through_real_owner_source_validation(agent_world, pinned_path_node):
    """With the pin's Node prepared (``pinned_path_node``: an isolated pin whose exact bundle is the PATH Node), the
    answered quiz carries the Connect through owner provenance, confirmation, the settings write and the audit."""
    from ouroboros import platform_layer
    from ouroboros.owner_quiz import record_answered, record_asked

    source = {"kind": "quiz", "task_id": "connect-task", "quiz_id": "connect-chrome"}
    record_asked(agent_world.drive, "connect-task", quiz_id="connect-chrome", options=["Connect", "Not now"],
                 question="Connect Chrome (shared tab)? Turning MCP on also starts GitHub.")
    unanswered, blocked = _agent_call(agent_world, owner_source=source, preview=True)
    assert blocked and unanswered["status_code"] == 403  # an asked but unanswered quiz is not owner intent
    record_answered(agent_world.drive, "connect-task", quiz_id="connect-chrome", option_index=0, request_id="answer-1")
    preview, blocked = _agent_call(agent_world, owner_source=source, preview=True)
    assert not blocked and preview["ready"] and preview["other_servers"] == [{"id": "github", "name": "GitHub"}]
    entry = agent_world.drive / "skills" / "external" / "chrome_browser" / "scripts" / "server.mjs"
    assert preview["servers"][0]["executable"]["args"] == [str(entry.resolve())]
    prepared = platform_layer.embedded_node_candidates(agent_world.drive.parent / "pinned-path-node")[0]
    assert os.path.normcase(preview["servers"][0]["executable"]["command"]) == os.path.normcase(str(prepared))
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
