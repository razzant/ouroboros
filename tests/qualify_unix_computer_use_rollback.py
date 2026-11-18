"""Two-version qualification on a disposable macOS runner with the MCP SDK installed.

Check out e758a219 alongside the candidate. From the candidate run
OUROBOROS_ROLLBACK_CHECKOUT=/path/to/old python -m pytest -q tests/qualify_unix_computer_use_rollback.py
Each phase starts a fresh interpreter on its own core against the same private data.
MCP discovery lists the connected window server. The only ordinary tool calls save and list a disabled SSH profile.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

OLD_CORE = "e758a2190298d9890b1f26e632f63775d0b33d55"
SKILL = "unix_computer_use"
SERVER = "mac_window_session"


@pytest.mark.parametrize("connection", ["never-connected", "saved-connection"])
def test_native_skill_across_rollback(tmp_path, connection):
    from ouroboros.test_environment import isolated_environment

    old = Path(os.environ["OUROBOROS_ROLLBACK_CHECKOUT"]).resolve(strict=True)
    driver = Path(__file__).resolve()
    new = driver.parents[1]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=old, text=True).strip() == OLD_CORE
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=old), "use a clean old checkout"
    for phase, repo in (("install", old), ("upgrade", new), ("rollback", old), ("restore", new)):
        env = isolated_environment(tmp_path, repo, source={"PATH": os.environ["PATH"]})
        env.update(OUROBOROS_RUNTIME_MODE="advanced", OUROBOROS_AGENT_PYTHON=sys.executable,
                   OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS="false", OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS="true")
        subprocess.run([sys.executable, "-I", "-B", str(driver), phase, str(tmp_path), connection],
                       cwd=repo, env=env, timeout=120, check=True)


def _check_ordinary_tools(scratch, phase):
    from ouroboros import extension_loader

    tools = extension_loader.snapshot()["tools"]
    screenshot = extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "screenshot"))
    arguments = screenshot["schema"]["properties"]
    assert "app" in arguments and ("window" in arguments) == (phase in {"upgrade", "restore"})
    baseline = scratch / "old-tools.json"
    if phase == "install":
        for name in ("screenshot", "element_act", "click", "key", "list_connections", "remote_exec"):
            assert extension_loader.extension_surface_name(SKILL, name) in tools
        baseline.write_text(json.dumps(tools), encoding="utf-8")
        add = extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "add_connection"))
        result = add["handler"](name="fixture", backend="ssh_macos", ssh_alias="fixture", enabled=False)
        assert json.loads(result)["ok"]
    else:
        assert set(json.loads(baseline.read_bytes())) <= set(tools)
    listing = extension_loader.get_tool(extension_loader.extension_surface_name(SKILL, "list_connections"))
    saved = json.loads(listing["handler"]())
    assert saved["active"] == "local" and saved["connections"]["fixture"] == {
        "backend": "ssh_macos", "ssh_alias": "fixture", "enabled": False, "active": False,
    }


def _phase(phase, scratch, connected):
    repo = Path.cwd()
    drive = scratch / "data"
    sys.path.insert(0, str(repo))  # -I excludes cwd and the candidate test directory
    from ouroboros import config, extension_loader, mcp_client, server_boot, skill_builtin, skill_mcp
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_loader import compute_content_hash, find_skill, save_enabled

    settings = drive / "settings.json"
    assert config.DATA_DIR == drive and config.SETTINGS_PATH == settings and config.REPO_DIR == repo
    assert Path(skill_builtin.__file__).resolve().parent == repo / "ouroboros"
    if phase == "install":
        settings.write_text('{"MCP_ENABLED":false,"MCP_SERVERS":[]}', encoding="utf-8")
        config.load_settings(migrate=True)  # settle the old core's ordinary settings migrations first
        installed = skill_builtin.install_builtin_skill(SKILL, drive_root=drive)
        assert installed["ok"], installed
        assert installed["auto_granted_keys"] == installed["auto_granted_permissions"] == []
        save_enabled(drive, SKILL, True, actor="owner_ui")
    if phase == "rollback":
        # This is the old parser reading the new installed bytes BEFORE its boot reseeds them.
        target = drive / "skills" / "native" / SKILL / "SKILL.md"
        incoming = parse_skill_manifest_text(target.read_text(encoding="utf-8"))
        assert incoming.version and not incoming.mcp_servers
        extra = incoming.raw_extra["optional_mcp_servers"]
        assert [s["id"] for s in extra] == [SERVER]
        mixed = parse_skill_manifest_text(json.dumps({
            "runtime": "node", "mcp_servers": [{"id": "existing", "entry": "server.mjs"}],
            "optional_mcp_servers": extra,
        }))
        assert [(s["id"], s["runtime"]) for s in mixed.mcp_servers] == [("existing", "node")]

    before = settings.read_bytes()
    count = skill_builtin.ensure_builtin_skills(drive, repo_dir=repo, dependency_pass=False)
    assert count == (0 if phase == "install" else 1), (phase, count)
    skill = find_skill(drive, SKILL)
    assert skill and skill.enabled and not skill.load_error
    assert skill.content_hash == compute_content_hash(repo / "skills" / SKILL)
    loaded = extension_loader.reload_all(drive, config.load_settings)
    assert loaded[SKILL] is None, loaded
    _check_ordinary_tools(scratch, phase)

    newer = phase in {"upgrade", "restore"}
    preview = skill_mcp.preview_connect(drive, SKILL)
    if newer:
        assert preview["ready"], preview
        server, = preview["servers"]
        assert server["id"] == SERVER and server["task_session"]
        assert server["change"] == ("unchanged" if connected and phase == "restore" else "add")
    else:
        assert preview["code"] == "no_mcp_servers", preview
    assert settings.read_bytes() == before
    if connected and phase == "upgrade":
        result = skill_mcp.connect(drive, SKILL, expected_content_hash=preview["content_hash"],
                                   settings_digest=preview["settings_digest"])
        assert result["ok"] and result["saved"], result
        before = settings.read_bytes()
        saved = json.loads(before)
        assert saved["MCP_ENABLED"] is True
        row, = saved["MCP_SERVERS"]
        assert row["skill"] == SKILL and row["id"] == SERVER and row["enabled"] and row["task_session"]
        assert row["transport"] == "stdio"
        assert not {"command", "args", "cwd"}.intersection(row)

    server_boot.mcp_startup()
    manager = mcp_client.get_manager()
    assert manager.discovery.wait(60), phase
    registered = {tool["name"] for tool in manager.list_tools_for_registry()}
    if not connected or phase == "install":
        assert manager.server_ids() == [] and not registered
    elif phase == "rollback":
        assert manager.server_ids() == [SERVER] and not registered
        assert manager.resolve_tool_name(f"mcp_{SERVER}__begin").status == "skill_unavailable"
        refused = manager.refresh_server(SERVER)
        assert refused["code"] == "MCP_SKILL_UNAVAILABLE" and refused["not_started"], refused
    else:
        assert registered == {f"mcp_{SERVER}__{tool}" for tool in ("begin", "frame", "release")}
    assert settings.read_bytes() == before


if __name__ == "__main__":
    _phase(sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3] == "saved-connection")
