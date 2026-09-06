"""Skill-widget surface regressions: preflight entry truth, reconcile receipts, catalogue liveness.

Single home for this stream's Python pins so no existing oversized test module
grows. No network, no real LLM calls; every case runs against an isolated
tmp_path drive root.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros.tools.registry import ToolContext


def _make_ctx(tmp_path: pathlib.Path) -> ToolContext:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    return ToolContext(repo_dir=repo_dir, drive_root=drive_root)


def _extension_manifest(name: str = "alpha", *, ui_tab: str = "") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: widget surface test\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [widget]\n"
        f"{ui_tab}"
        "---\n"
        "body\n"
    )


def _module_ui_tab(entry: str = "widget.js") -> str:
    return (
        "ui_tab:\n"
        "  id: main\n"
        "  title: Main\n"
        "  render:\n"
        "    kind: module\n"
        f"    entry: {entry}\n"
        "    height: 480\n"
    )


def _make_skill(tmp_path: pathlib.Path, monkeypatch, manifest: str, plugin: str) -> pathlib.Path:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = skills_root / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    (skill_dir / "plugin.py").write_text(plugin, encoding="utf-8")
    return skill_dir


_TRIVIAL_PLUGIN = "def register(api):\n    pass\n"


# --------------------------------------------------------------------------- F12


def test_skill_preflight_flags_missing_module_widget_entry(tmp_path, monkeypatch):
    """A declared module entry that is not on disk must not read as verified ok."""
    ctx = _make_ctx(tmp_path)
    skill_dir = _make_skill(
        tmp_path,
        monkeypatch,
        _extension_manifest(ui_tab=_module_ui_tab()),
        _TRIVIAL_PLUGIN,
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))
    entry_rows = [row for row in result["widgets"] if row["item"] == "widget_entry_exists"]
    assert entry_rows, result["widgets"]
    assert result["ok"] is False
    assert entry_rows[0]["ok"] is False
    assert "widget.js" in entry_rows[0]["detail"]
    assert entry_rows[0]["source"] == "manifest.ui_tab.render"

    # Same declaration, file now present: the row flips and names the entry.
    (skill_dir / "widget.js").write_text("const a = 1;\n", encoding="utf-8")
    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))
    entry_rows = [row for row in result["widgets"] if row["item"] == "widget_entry_exists"]
    assert entry_rows[0]["ok"] is True
    assert entry_rows[0]["detail"] == "widget.js"


def test_skill_preflight_checks_plugin_registered_module_entry(tmp_path, monkeypatch):
    """The plugin.py register_ui_tab path is covered without touching the AST walker."""
    ctx = _make_ctx(tmp_path)
    _make_skill(
        tmp_path,
        monkeypatch,
        _extension_manifest(),
        "_UI_RENDER = {'kind': 'module', 'entry': 'missing.js', 'height': 400}\n"
        "def register(api):\n"
        "    api.register_ui_tab('main', 'Main', render=_UI_RENDER)\n",
    )

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))
    entry_rows = [row for row in result["widgets"] if row["item"] == "widget_entry_exists"]
    assert entry_rows, result["widgets"]
    assert result["ok"] is False
    assert entry_rows[0]["ok"] is False
    assert "missing.js" in entry_rows[0]["detail"]
    assert entry_rows[0]["source"].startswith("plugin.py:")


def test_noncanonical_module_entry_fails_preflight_and_runtime_validation(
    tmp_path, monkeypatch
):
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.extension_ui_validation import validate_ui_render
    from ouroboros.tools import skill_preflight as sp

    ctx = _make_ctx(tmp_path)
    _make_skill(
        tmp_path,
        monkeypatch,
        _extension_manifest(ui_tab=_module_ui_tab("widget space.js")),
        _TRIVIAL_PLUGIN,
    )

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))
    rows = [row for row in result["widgets"] if row["item"] == "widget_schema"]
    assert result["ok"] is False
    assert rows[0]["ok"] is False
    assert rows[0]["verified"] is True
    assert "browser-safe" in rows[0]["detail"]
    with pytest.raises(ExtensionRegistrationError, match="browser-safe"):
        validate_ui_render({"kind": "module", "entry": "widget space.js"})


# --------------------------------------------------------------------------- F13


@pytest.mark.parametrize(
    ("source", "expect_block"),
    [("export const a = 1;\n", True), ("const a = 1;\n", False)],
)
def test_skill_preflight_parses_module_entry_as_classic_script(
    tmp_path, monkeypatch, source, expect_block
):
    """The declared entry is checked in the grammar the frame actually runs it in."""
    ctx = _make_ctx(tmp_path)
    skill_dir = _make_skill(
        tmp_path,
        monkeypatch,
        _extension_manifest(ui_tab=_module_ui_tab()),
        _TRIVIAL_PLUGIN,
    )
    (skill_dir / "widget.js").write_text(source, encoding="utf-8")

    from ouroboros.tools import skill_preflight as sp

    result = json.loads(sp._handle_skill_preflight(ctx, skill="alpha"))
    rows = [row for row in result["files"] if row["path"] == "widget.js"]
    assert rows, result["files"]
    row = rows[0]
    assert row["grammar"] == "classic_script"
    if row.get("skipped"):
        # No usable node runtime: the honest skip branch, never a verdict. The
        # grammar assertion above still held, so the selection is covered.
        assert row["skip_reason"] in {"runtime_unavailable", "validator_killed", "validator_timeout"}
        assert result["degraded"] is True
        pytest.skip(f"no usable node runtime for the classic-script check ({row['skip_reason']})")
    if expect_block:
        assert row["ok"] is False
        assert result["ok"] is False
    else:
        assert row["ok"] is True


def test_skill_preflight_validator_env_keeps_windows_process_base_keys(tmp_path, monkeypatch):
    """The scrubbed validator env still lets a Windows child start.

    A node started without SystemRoot aborts before it reads the script, so the
    valid entry above read as a syntax error on windows-latest (7.0.0-rc.9).
    Pinned on every host by simulating that environment: the process-base keys
    are forwarded, everything else stays scrubbed, a POSIX env is byte-identical
    to before, and the validator's pipes stay BYTES decoded as UTF-8 with
    replacement -- the 0x8f that kills a locale-decoded (cp1252) reader thread
    is inert here.
    """
    from ouroboros.tools import skill_preflight as sp

    seen: dict = {}

    class _FakeProc:
        returncode = 0
        pid = 4242

        def communicate(self, timeout=None):
            return b"", b"\x8f\xff not utf-8"

    def _fake_popen(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(sp, "Popen", _fake_popen)
    for key in sp._WINDOWS_BASE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-the-validator")

    result = sp._run_check(["node", "--check", "widget.js"], cwd=tmp_path)
    posix_env = seen["kwargs"]["env"]
    assert set(posix_env) == {"PATH", "HOME", "LANG"}
    assert posix_env["LANG"] == "C.UTF-8"
    assert not any(key in seen["kwargs"] for key in ("text", "universal_newlines", "encoding"))
    assert result["returncode"] == 0
    assert result["stderr"] == "\ufffd\ufffd not utf-8"

    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")
    monkeypatch.setenv("TEMP", "C:\\Users\\runneradmin\\AppData\\Local\\Temp")
    sp._run_check(["node", "--check", "widget.js"], cwd=tmp_path)
    windows_env = seen["kwargs"]["env"]
    assert windows_env["SYSTEMROOT"] == "C:\\Windows"
    assert windows_env["TEMP"] == "C:\\Users\\runneradmin\\AppData\\Local\\Temp"
    assert set(windows_env) == {"PATH", "HOME", "LANG", "SYSTEMROOT", "TEMP"}
    assert "OPENROUTER_API_KEY" not in windows_env


# ------------------------------------------------------------- S1-07 / S1-02 / F14


def _prepare_live_extension(tmp_path: pathlib.Path, name: str = "extlive"):
    """Write, enable and PASS-review one extension so the loader will accept it."""
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Live extension.\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        'permissions: ["tool"]\n'
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n"
        "    api.register_tool('ping', lambda ctx: 'pong', description='Ping.', schema={})\n",
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, name, True, actor="test_fixture")
    save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


@pytest.fixture(autouse=True)
def _clean_loader_state(monkeypatch):
    from tests._shared import clean_extension_runtime_state

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()



def _set_process_role(monkeypatch, server: bool) -> None:
    """Pin which process "answers" a reconcile.

    v7 split: the loader stamps its own receipts through its imported binding,
    while the reconcile queue and the liveness projection read the OWNER
    (``extension_companion.is_server_process``) at call time — so the pin must
    land on both, or the receipt vocabulary (``requested`` / ``request_failed``)
    is decided by whatever process-pid state an earlier test left behind."""
    import ouroboros.extension_companion as extension_companion
    import ouroboros.extension_loader as extension_loader

    monkeypatch.setattr(extension_companion, "is_server_process", lambda: server)
    monkeypatch.setattr(extension_loader, "is_server_process", lambda: server)

def test_reconcile_receipt_names_the_answering_process(tmp_path, monkeypatch):
    """A reconcile receipt says which process answered and whether the marker landed."""
    from ouroboros import extension_loader

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)

    _set_process_role(monkeypatch, True)
    state = extension_loader.reconcile_extension(
        loaded.name, drive_root, lambda: {}, repo_path=str(repo_root)
    )
    assert state["process"] == "server"
    assert state["server_reconcile"] == ""

    extension_loader.unload_extension(loaded.name)
    _set_process_role(monkeypatch, False)
    state = extension_loader.reconcile_extension(
        loaded.name, drive_root, lambda: {}, repo_path=str(repo_root)
    )
    assert state["process"] == "worker"
    assert state["server_reconcile"] == "requested"
    assert list((drive_root / "state" / "extension_reconcile").glob("*")), "no marker written"


def test_reconcile_records_health_for_the_resulting_runtime_state(tmp_path, monkeypatch):
    """Each reconcile receipt updates the durable health projection."""
    from ouroboros import extension_health, extension_loader
    from ouroboros.skill_loader import save_enabled

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)
    _set_process_role(monkeypatch, True)

    state = extension_loader.reconcile_extension(
        loaded.name, drive_root, lambda: {}, repo_path=str(repo_root)
    )
    assert state["live_loaded"] is True
    assert (extension_health.read_extension_health(drive_root, loaded.name) or {})["status"] == "live"

    save_enabled(drive_root, loaded.name, False, actor="test_fixture")
    state = extension_loader.reconcile_extension(
        loaded.name, drive_root, lambda: {}, repo_path=str(repo_root)
    )
    assert state["desired_live"] is False
    assert (extension_health.read_extension_health(drive_root, loaded.name) or {})["status"] == "inactive"


def test_reconcile_receipt_reports_a_failed_marker_request(tmp_path, monkeypatch):
    """A failed worker handoff cannot replace the authoritative server health."""
    from ouroboros import extension_health, extension_loader
    from ouroboros import extension_reconcile_queue

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)
    extension_health.record_extension_health(
        drive_root, loaded.name, status="live", version="0.0.1", sha="server-good",
    )
    extension_health.record_extension_health(
        drive_root, loaded.name, status="broken", version="0.0.2", sha="shared-sha",
        reason="load_error", load_error="server import failed",
    )
    _set_process_role(monkeypatch, False)

    def boom(*_a, **_k):
        raise OSError("marker directory is read-only")

    monkeypatch.setattr(extension_reconcile_queue, "request_extension_reconcile", boom)
    state = extension_loader.reconcile_extension(
        loaded.name, drive_root, lambda: {}, repo_path=str(repo_root),
        health_stamp=("0.0.2", "shared-sha"),
    )
    health = extension_health.read_extension_health(drive_root, loaded.name) or {}
    server = health.get("last_observed") or {}
    worker = (health.get("observations") or {}).get("worker") or {}

    assert state["process"] == "worker"
    assert state["server_reconcile"] == "request_failed"
    assert state["action"] == "extension_loaded"
    assert health["status"] == "broken"
    assert health["regressed"] is True
    assert health["last_known_good"]["sha"] == "server-good"
    assert server["status"] == "broken"
    assert server["sha"] == "shared-sha"
    assert worker["status"] == "live"
    assert worker["sha"] == "shared-sha"
    assert worker["server_reconcile"] == "request_failed"
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_k: object())
    monkeypatch.setattr("ouroboros.skill_loader.load_enabled", lambda *_a, **_k: True)
    assert [row["skill"] for row in extension_health.regressed_extensions(drive_root)] == [
        loaded.name
    ]


def test_toggle_skill_receipt_carries_process_and_marker_outcome(tmp_path, monkeypatch):
    """The agent-facing toggle receipt is where the 'already_live/ready' misreading happened."""
    from ouroboros.tools import skill_exec as skill_exec_mod

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    ctx = ToolContext(repo_dir=tmp_path / "repo2", drive_root=drive_root)
    (tmp_path / "repo2").mkdir()

    payload = json.loads(skill_exec_mod._handle_toggle_skill(ctx, skill=loaded.name, enabled=True))
    assert payload["process"] in {"server", "worker"}
    assert payload["server_reconcile"] in {"", "requested", "request_failed"}


def test_save_enabled_appends_one_typed_actor_row(tmp_path):
    """Enablement changes leave a durable, non-rotating, actor-attributed record."""
    from ouroboros.skill_loader import save_enabled

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    save_enabled(drive_root, "alpha", True, actor="owner_ui", reason="client_host=127.0.0.1")
    save_enabled(drive_root, "alpha", False, actor="agent_tool", reason="task-7")
    save_enabled(drive_root, "alpha", True)

    rows = [
        json.loads(line)
        for line in (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("type") == "skill_enabled_changed"]
    assert [(r["enabled"], r["previous"], r["actor"]) for r in rows] == [
        (True, False, "owner_ui"),
        (False, True, "agent_tool"),
        (True, False, ""),
    ]
    assert rows[0]["reason"] == "client_host=127.0.0.1"


def test_save_enabled_row_is_disclosure_never_a_gate(tmp_path):
    """An unwritable logs path must not fail the enablement write."""
    from ouroboros.skill_loader import load_enabled, save_enabled

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    (drive_root / "logs").write_text("not a directory", encoding="utf-8")
    save_enabled(drive_root, "alpha", True, actor="owner_ui")
    assert load_enabled(drive_root, "alpha") is True


def test_save_enabled_best_effort_disclosure_contract_is_documented():
    from ouroboros.skill_loader import save_enabled

    architecture = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "ARCHITECTURE.md").read_text(
        encoding="utf-8"
    )
    assert "best-effort" in str(save_enabled.__doc__)
    assert "append failure is logged and never blocks the enablement change" in architecture


def test_api_skill_toggle_records_the_owner_ui_actor(tmp_path, monkeypatch):
    """The HTTP owner toggle labels itself, so the incident class is reconstructible."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state
    from tests.test_extensions_api import _make_client, _stop_patches, _write_ext

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_actor",
        permissions=["tool"],
        plugin="def register(api):\n    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, _patches = _make_client(tmp_path, monkeypatch)
    try:
        save_review_state(
            drive_root,
            "ext_actor",
            SkillReviewState(
                status="pass",
                content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            ),
        )
        resp = client.post("/api/skills/ext_actor/toggle", json={"enabled": True})
        assert resp.status_code == 200, resp.text
    finally:
        client.close()
        _stop_patches(_patches)   # the started patches (the password resolver among them) must not outlive the test

    rows = [
        json.loads(line)
        for line in (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("type") == "skill_enabled_changed" and r.get("skill") == "ext_actor"]
    assert rows, "the owner toggle left no enablement row"
    assert rows[-1]["actor"] == "owner_ui"
    assert rows[-1]["enabled"] is True
    assert rows[-1]["reason"].startswith("client_host=")


def test_summarize_skills_projects_live_extension_facts(tmp_path, monkeypatch):
    """The catalogue reports the same live facts as /api/extensions, not worker guesses."""
    from ouroboros import skill_loader
    from ouroboros.skill_loader import summarize_skills

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    calls = {"n": 0}
    real_discover = skill_loader.discover_skills

    def counting(*a, **k):
        calls["n"] += 1
        return real_discover(*a, **k)

    monkeypatch.setattr(skill_loader, "discover_skills", counting)
    summary = summarize_skills(drive_root)
    # One walk for the catalogue itself; `skills=` must stop the per-row re-walk.
    assert calls["n"] == 1, f"discover_skills ran {calls['n']} times"

    row = next(r for r in summary["skills"] if r["name"] == loaded.name)
    assert row["desired_live"] is True
    assert row["live_loaded"] is False
    assert row["live_reason"]
    assert row["process"] in {"server", "worker"}
    assert row["available_for_execution"] is False


def test_skill_exec_extension_message_reports_typed_liveness(tmp_path, monkeypatch):
    """skill_exec no longer asserts register(api) ran for an extension that never loaded."""
    from ouroboros.tools import skill_exec as skill_exec_mod

    loaded, repo_root, drive_root = _prepare_live_extension(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    ctx = ToolContext(repo_dir=tmp_path / "repo3", drive_root=drive_root)
    (tmp_path / "repo3").mkdir()

    out = skill_exec_mod._handle_skill_exec(ctx, skill=loaded.name, script="x.py")
    assert "SKILL_EXEC_EXTENSION" in out
    assert "live_loaded=False" in out
    assert "has already been called" not in out
