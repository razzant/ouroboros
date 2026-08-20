"""Characterization matrix for the PluginAPI load / dispatch / unload lifecycle.

Pins the observable behaviour of the extension runtime: which permission each
registration method demands and the exact sentence it refuses with, the
provider-safe names and registry payloads a registration produces, which
capabilities an out-of-process child loses, what the settings reader discloses,
and what unload tears down. Every assertion is a fact about the PluginAPI
contract rather than about which module happens to define a helper, so the file
holds across an owner split of ``ouroboros/extension_loader.py``.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from ouroboros import extension_loader
from ouroboros.contracts.plugin_api import (
    ALWAYS_AVAILABLE_CAPABILITIES,
    MATRIX_CAPABILITIES,
    ExecutionMode,
    ExtensionRegistrationError,
    PluginAPI,
)
from ouroboros.extension_companion import init_server_process_pid
from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state
from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _pristine_extension_runtime(monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _api(tmp_path: pathlib.Path, permissions, **overrides):
    config = extension_loader._PluginAPIConfig(
        skill_name=overrides.pop("skill_name", "demo"),
        permissions=list(permissions),
        env_allowlist=list(overrides.pop("env_allowlist", [])),
        state_dir=tmp_path / "state",
        settings_reader=overrides.pop("settings_reader", lambda: {}),
        **overrides,
    )
    return extension_loader.PluginAPIImpl(config)


def _write_extension(tmp_path: pathlib.Path, name: str, body: str, permissions: list[str]):
    """Write, enable and PASS-review one extension so the loader accepts it."""
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(parents=True, exist_ok=True)
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: matrix fixture.\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        f"permissions: {list(permissions)!r}\n"
        "env_from_settings: []\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(body, encoding="utf-8")
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, name, True)
    save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


# (method name, call, declared permission). One row per permission-gated PluginAPI verb.
_PERMISSION_MATRIX = (
    ("register_tool", lambda api: api.register_tool("t", lambda: "", description="d", schema={}), "tool"),
    ("register_route", lambda api: api.register_route("p", lambda: None), "route"),
    ("register_ws_handler", lambda api: api.register_ws_handler("m", lambda: None), "ws_handler"),
    ("register_ui_tab", lambda api: api.register_ui_tab("tab", "Tab"), "widget"),
    ("register_settings_section", lambda api: api.register_settings_section("s", "S", schema={}), "widget"),
    ("register_supervised_task", lambda api: api.register_supervised_task("job", lambda: None), "supervised_task"),
    ("register_companion_process", lambda api: api.register_companion_process("daemon"), "companion_process"),
    ("subscribe_event", lambda api: api.subscribe_event("topic", lambda _e: None), "subscribe_event"),
    ("send_ws_message", lambda api: api.send_ws_message("m", {}), "ws_handler"),
)


@pytest.mark.parametrize("method,call,permission", _PERMISSION_MATRIX, ids=[row[0] for row in _PERMISSION_MATRIX])
def test_every_registration_verb_refuses_without_its_declared_permission(tmp_path, method, call, permission):
    """A missing permission is refused by name, and the refusal names what the
    manifest actually declared — never a silent no-op."""
    api = _api(tmp_path, permissions=[])
    with pytest.raises(ExtensionRegistrationError) as excinfo:
        call(api)
    assert str(excinfo.value) == f"skill 'demo' cannot {permission!r} — manifest permissions=[]"


def test_the_permission_vocabulary_is_closed():
    api = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="demo", permissions=["not_a_permission"], env_allowlist=[],
        state_dir=pathlib.Path("/nonexistent"), settings_reader=lambda: {},
    ))
    with pytest.raises(ExtensionRegistrationError) as excinfo:
        api._require("not_a_permission")
    assert str(excinfo.value) == "unknown extension permission 'not_a_permission'"


def test_registered_surfaces_are_namespaced_and_snapshot_exactly(tmp_path):
    api = _api(tmp_path, permissions=["tool", "route", "ws_handler", "widget"])
    api.register_tool("echo", lambda ctx: "ok", description="Echo", schema={"type": "object"}, timeout_sec=7)
    api.register_route("status", lambda: None, methods=["get", "post", "get"])
    api.register_ws_handler("ping", lambda: None)
    api.register_ui_tab("panel", "Panel", render={"span": 5})
    api.register_settings_section("prefs", "Prefs", schema={})

    prefix = extension_loader.extension_name_prefix("demo")
    assert prefix == "ext_6_r_demo_"
    assert extension_loader.parse_extension_surface_name(f"{prefix}echo") == ("r_demo", "echo")

    snapshot = extension_loader.snapshot()
    assert snapshot["extensions"] == ["demo"]
    assert snapshot["tools"] == [f"{prefix}echo"]
    assert snapshot["routes"] == ["/api/extensions/demo/status"]
    assert snapshot["ws_handlers"] == [f"{prefix}ping"]
    assert snapshot["ui_tabs_pending"] == []
    assert [tab["key"] for tab in snapshot["ui_tabs"]] == ["demo:panel"]
    assert [section["key"] for section in snapshot["settings_sections"]] == ["demo:prefs"]

    tool = extension_loader.get_tool(f"{prefix}echo")
    assert tool is not None
    assert tool["name"] == f"{prefix}echo"
    assert tool["description"] == "Echo"
    assert tool["schema"] == {"type": "object"}
    assert tool["timeout_sec"] == 7
    assert tool["skill"] == "demo"
    assert tool["wants_ctx"] is True
    assert callable(tool["_model_credential_probe"])
    assert extension_loader.get_tool("ext_6_r_demo_missing") is None

    route = extension_loader.list_routes()["/api/extensions/demo/status"]
    assert route["methods"] == ("GET", "POST")
    assert route["skill"] == "demo"
    assert set(extension_loader.list_ws_handlers()) == {f"{prefix}ping"}

    # The widget span is normalised to the two-column grid, not passed through.
    tab = snapshot["ui_tabs"][0]
    assert (tab["span"], tab["grid_span"], tab["ws_prefix"]) == (2, 2, prefix)
    assert tab["ui_host_pending"] is True


@pytest.mark.parametrize(
    "call,message_head",
    (
        (lambda api: api.register_tool("", lambda: "", description="", schema={}), "tool name must be non-empty"),
        (lambda api: api.register_tool("x" * 25, lambda: "", description="", schema={}),
         "tool name must be <= 24 characters"),
        (lambda api: api.register_tool("bad name", lambda: "", description="", schema={}),
         "tool name must be alnum/underscore only"),
        (lambda api: api.register_route("", lambda: None), "path must be non-empty"),
        (lambda api: api.register_route("/abs", lambda: None), "path must be relative, not absolute"),
        (lambda api: api.register_route("a/../b", lambda: None), "path must not contain '..' segments"),
        (lambda api: api.register_route("ok", lambda: None, methods=[]), "route methods must be non-empty"),
        (lambda api: api.register_route("ok", lambda: None, methods=["TRACE"]), "route methods ['TRACE'] are unsupported"),
    ),
)
def test_surface_names_and_route_methods_are_validated_at_registration(tmp_path, call, message_head):
    api = _api(tmp_path, permissions=["tool", "route"])
    with pytest.raises(ExtensionRegistrationError) as excinfo:
        call(api)
    assert str(excinfo.value).startswith(message_head)


def test_a_duplicate_surface_key_is_refused(tmp_path):
    api = _api(tmp_path, permissions=["tool"])
    api.register_tool("echo", lambda ctx: "", description="", schema={})
    with pytest.raises(ExtensionRegistrationError) as excinfo:
        api.register_tool("echo", lambda ctx: "", description="", schema={})
    assert str(excinfo.value) == "tool 'ext_6_r_demo_echo' already registered"


def test_out_of_process_children_lose_exactly_the_matrix_capabilities(tmp_path, monkeypatch):
    """The contract matrix is the authority; the loader refuses exactly what it
    marks unavailable, and says which capabilities remain."""
    monkeypatch.setenv("OUROBOROS_EXTENSION_PROCESS_CHILD", "1")
    assert extension_loader.current_execution_mode() is ExecutionMode.OUT_OF_PROCESS
    api = _api(tmp_path, permissions=sorted({"subscribe_event", "supervised_task", "companion_process", "ws_handler"}))
    for capability, call in (
        ("subscribe_event", lambda: api.subscribe_event("t", lambda _e: None)),
        ("register_supervised_task", lambda: api.register_supervised_task("job", lambda: None)),
    ):
        with pytest.raises(ExtensionRegistrationError) as excinfo:
            call()
        assert str(excinfo.value) == (
            f"{capability} is not available to out-of-process (isolated-dep) extensions "
            "in the per-call child; declare a companion_process for long-running work "
            "and host-event subscription. Available capabilities here: "
            "on_unload, register_companion_process, register_route, register_settings_section, "
            "register_tool, register_ui_tab, register_ws_handler, send_ws_message."
        )
    # on_unload and send_ws_message stay available: neither raises the child refusal.
    api.on_unload(lambda: None)
    api.send_ws_message("m", {})
    runtime = api.get_runtime_info()
    assert runtime["execution_mode"] == "out_of_process"
    assert runtime["capabilities"] == sorted(MATRIX_CAPABILITIES - {"subscribe_event", "register_supervised_task"})

    monkeypatch.delenv("OUROBOROS_EXTENSION_PROCESS_CHILD")
    assert extension_loader.current_execution_mode() is ExecutionMode.IN_PROCESS
    assert _api(tmp_path, permissions=[]).get_runtime_info()["capabilities"] == sorted(MATRIX_CAPABILITIES)


def test_the_plugin_api_surface_is_fully_classified_and_structurally_satisfied(tmp_path):
    api = _api(tmp_path, permissions=[])
    assert isinstance(api, PluginAPI)
    for name in MATRIX_CAPABILITIES | ALWAYS_AVAILABLE_CAPABILITIES:
        assert callable(getattr(api, name)), name


def test_get_settings_discloses_only_allowlisted_and_granted_keys(tmp_path):
    settings = {"EXT_KEY": "visible", "OTHER": "hidden", "OPENROUTER_API_KEY": "secret"}
    api = _api(
        tmp_path,
        permissions=["read_settings"],
        env_allowlist=["EXT_KEY", "OPENROUTER_API_KEY"],
        settings_reader=lambda: settings,
    )
    assert api.get_settings(["EXT_KEY", "OTHER", "OPENROUTER_API_KEY"]) == {"EXT_KEY": "visible"}

    granted = _api(
        tmp_path,
        permissions=["read_settings"],
        env_allowlist=["OPENROUTER_API_KEY"],
        settings_reader=lambda: settings,
        granted_keys=["OPENROUTER_API_KEY"],
    )
    assert granted.get_settings(["OPENROUTER_API_KEY"]) == {"OPENROUTER_API_KEY": "secret"}

    # Without the permission the reader fails closed and leaks no key presence.
    unprivileged = _api(tmp_path, permissions=[], env_allowlist=["EXT_KEY"], settings_reader=lambda: settings)
    assert unprivileged.get_settings(["EXT_KEY"]) == {}


def test_state_dir_and_job_dir_are_bound_to_the_skill_state_root(tmp_path):
    api = _api(tmp_path, permissions=[])
    assert api.get_state_dir() == str(tmp_path / "state")
    job = api.skill_job_dir("../escape me")
    assert job.parent == tmp_path / "state" / "jobs"
    assert job.name.startswith("escape_me-")
    assert sorted(child.name for child in job.iterdir()) == ["assets", "output", "tmp"]
    assert api.skill_job_dir("../escape me") == job


def test_ws_broadcast_reaches_the_installed_host_broadcaster(tmp_path):
    sent: list[dict] = []
    extension_loader.set_ws_broadcaster(sent.append)
    api = _api(tmp_path, permissions=["ws_handler"])
    api.send_ws_message("ping", {"n": 1})
    assert sent == [{"type": "ext_6_r_demo_ping", "data": {"n": 1}, "skill": "demo"}]
    # A dropped broadcaster silently stops delivery instead of raising into the extension.
    extension_loader.set_ws_broadcaster(None)
    api.send_ws_message("ping", {"n": 2})
    assert len(sent) == 1


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (lambda drive, name: save_enabled(drive, name, False), "skill 'gate' is disabled"),
        (
            lambda drive, name: save_review_state(drive, name, SkillReviewState(status="fail", content_hash="")),
            "skill 'gate' must carry a fresh executable review",
        ),
    ),
)
def test_load_refuses_a_disabled_or_unreviewed_extension(tmp_path, mutation, expected):
    loaded, repo_root, drive_root = _write_extension(tmp_path, "gate", "def register(api):\n    pass\n", [])
    mutation(drive_root, "gate")
    loaded = find_skill(drive_root, "gate", repo_path=str(repo_root))
    error = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, repo_path=str(repo_root))
    assert error is not None and error.startswith(expected)
    assert extension_loader.snapshot()["extensions"] == []


def test_load_requires_an_explicit_drive_root(tmp_path):
    loaded, _repo_root, _drive_root = _write_extension(tmp_path, "noroot", "def register(api):\n    pass\n", [])
    with pytest.raises(TypeError) as excinfo:
        extension_loader.load_extension(loaded, lambda: {})
    assert str(excinfo.value) == "load_extension requires explicit drive_root"


def test_load_refuses_a_plugin_without_a_register_callable(tmp_path):
    loaded, repo_root, drive_root = _write_extension(tmp_path, "noreg", "VALUE = 1\n", [])
    error = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, repo_path=str(repo_root))
    assert error == "skill 'noreg' plugin.py does not export a register(api) callable"
    assert extension_loader.snapshot()["extensions"] == []


def test_a_registration_error_during_register_tears_the_partial_load_down(tmp_path):
    body = (
        "def register(api):\n"
        "    api.register_tool('ok', lambda ctx: '', description='', schema={})\n"
        "    api.register_route('p', lambda: None)\n"
    )
    loaded, repo_root, drive_root = _write_extension(tmp_path, "partial", body, ["tool"])
    error = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, repo_path=str(repo_root))
    assert error == (
        "skill 'partial' registration error: skill 'partial' cannot 'route' "
        "— manifest permissions=['tool']"
    )
    assert extension_loader.snapshot() == {
        "extensions": [], "tools": [], "routes": [], "ws_handlers": [],
        "ui_tabs": [], "ui_tabs_pending": [], "settings_sections": [],
    }


def test_the_load_dispatch_unload_cycle_installs_then_removes_every_surface(tmp_path):
    init_server_process_pid()
    body = (
        "STATE = {'unloaded': 0}\n"
        "def _echo(ctx, text=''):\n"
        "    return 'echo:' + str(text)\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='Echo', schema={})\n"
        "    api.register_route('status', lambda: None)\n"
        "    api.register_ws_handler('ping', lambda: None)\n"
        "    api.register_ui_tab('panel', 'Panel')\n"
        "    api.on_unload(lambda: STATE.__setitem__('unloaded', 1))\n"
    )
    loaded, repo_root, drive_root = _write_extension(
        tmp_path, "cycle", body, ["tool", "route", "ws_handler", "widget"],
    )
    assert extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, repo_path=str(repo_root),
    ) is None

    prefix = extension_loader.extension_name_prefix("cycle")
    snapshot = extension_loader.snapshot()
    assert snapshot["extensions"] == ["cycle"]
    assert snapshot["tools"] == [f"{prefix}echo"]
    assert extension_loader.is_extension_live("cycle", drive_root, repo_path=str(repo_root)) is True

    # Dispatch runs the registered handler through the loader's runtime wrapper.
    entry = extension_loader.get_tool(f"{prefix}echo")
    assert entry["wants_ctx"] is True
    assert entry["handler"](None, text="hi") == "echo:hi"
    module_key = extension_loader._module_key("cycle")
    module = sys.modules[module_key]
    assert module.STATE == {"unloaded": 0}

    extension_loader.unload_extension("cycle")

    assert module.STATE == {"unloaded": 1}
    assert module_key not in sys.modules
    assert extension_loader.snapshot() == {
        "extensions": [], "tools": [], "routes": [], "ws_handlers": [],
        "ui_tabs": [], "ui_tabs_pending": [], "settings_sections": [],
    }
    assert extension_loader.list_routes() == {}
    assert extension_loader.list_ws_handlers() == {}
    assert extension_loader.list_companion_names() == []
    assert extension_loader.is_extension_live("cycle", drive_root, repo_path=str(repo_root)) is False


def test_reconcile_walks_the_load_already_live_unload_states(tmp_path):
    init_server_process_pid()
    loaded, repo_root, drive_root = _write_extension(
        tmp_path, "recon",
        "def register(api):\n    api.register_tool('echo', lambda ctx: '', description='', schema={})\n",
        ["tool"],
    )
    first = extension_loader.reconcile_extension("recon", drive_root, lambda: {}, repo_path=str(repo_root))
    assert (first["action"], first["reason"], first["live_loaded"]) == ("extension_loaded", "ready", True)

    second = extension_loader.reconcile_extension("recon", drive_root, lambda: {}, repo_path=str(repo_root))
    assert second["action"] == "extension_already_live"

    save_enabled(drive_root, "recon", False)
    third = extension_loader.reconcile_extension("recon", drive_root, lambda: {}, repo_path=str(repo_root))
    assert (third["action"], third["reason"], third["live_loaded"]) == ("extension_unloaded", "disabled", False)

    missing = extension_loader.reconcile_extension("ghost", drive_root, lambda: {}, repo_path=str(repo_root))
    assert (missing["action"], missing["reason"]) == ("extension_inactive", "missing")


def test_runtime_state_reports_the_liveness_authority_for_an_unknown_skill(tmp_path):
    _loaded, repo_root, drive_root = _write_extension(tmp_path, "known", "def register(api):\n    pass\n", [])
    state = extension_loader.runtime_state_for_skill_name("ghost", drive_root, repo_path=str(repo_root))
    assert state == {
        "skill": "ghost", "type": "extension", "runtime_mode": "", "enabled": False,
        "review_status": "missing", "review_stale": True, "load_error": "skill not found",
        "desired_live": False, "live_loaded": False, "loaded_present": False,
        "loaded_matches_current": False, "reason": "missing",
    }


def test_reload_all_reports_one_entry_per_extension(tmp_path):
    init_server_process_pid()
    loaded, repo_root, drive_root = _write_extension(
        tmp_path, "bulk",
        "def register(api):\n    api.register_tool('echo', lambda ctx: '', description='', schema={})\n",
        ["tool"],
    )
    assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root)) == {"bulk": None}
    assert extension_loader.snapshot()["extensions"] == ["bulk"]

    save_enabled(drive_root, "bulk", False)
    assert extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root)) == {"bulk": "disabled"}
    assert extension_loader.snapshot()["extensions"] == []
