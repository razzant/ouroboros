"""Out-of-process extension parity (v6.15.0): capability matrix + negotiation,
on_unload at teardown, companion-as-cataloged-surface, and the durable health diff.

These tests pin the new contract so a future change cannot silently re-narrow the
out-of-process PluginAPI (the exact regression class behind the anime_studio
on_unload failure) or drop the live->broken health memory.
"""
from __future__ import annotations


from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, extension_health
from ouroboros.contracts.plugin_api import (
    ALWAYS_AVAILABLE_CAPABILITIES,
    MATRIX_CAPABILITIES,
    OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES,
    ExecutionMode,
    ExtensionRegistrationError,
    PluginAPI,
    available_capabilities,
    capability_available,
)
from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clear_loader_state():
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _public_plugin_api_methods() -> set[str]:
    return {
        m for m in dir(PluginAPI)
        if not m.startswith("_") and callable(getattr(PluginAPI, m, None))
    }


# --- contract: matrix shape + coverage --------------------------------------


def test_matrix_capabilities_are_real_plugin_api_methods():
    methods = _public_plugin_api_methods()
    assert MATRIX_CAPABILITIES <= methods
    assert OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES <= MATRIX_CAPABILITIES


def test_matrix_plus_always_available_covers_whole_surface():
    """A new PluginAPI method cannot be added without classifying it."""
    methods = _public_plugin_api_methods()
    classified = set(MATRIX_CAPABILITIES) | set(ALWAYS_AVAILABLE_CAPABILITIES)
    assert methods == classified, (
        f"Unclassified PluginAPI methods: {methods - classified}; "
        f"stale matrix entries: {classified - methods}"
    )


def test_capability_available_per_mode():
    for cap in MATRIX_CAPABILITIES:
        assert capability_available(cap, ExecutionMode.IN_PROCESS) is True
    assert available_capabilities(ExecutionMode.IN_PROCESS) == MATRIX_CAPABILITIES
    oop = available_capabilities(ExecutionMode.OUT_OF_PROCESS)
    assert oop == MATRIX_CAPABILITIES - OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES
    # Realtime providers and the call feature hold sockets and tasks for a call's lifetime; a companion
    # refresh belongs to the server process that supervises it, never a per-call child.
    assert OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES == {
        "subscribe_event", "register_supervised_task", "register_realtime_provider", "register_live_conversation",
        "refresh_companion_process",
    }
    assert {"on_unload", "send_ws_message", "register_companion_process"} <= oop


# --- guard <-> matrix lockstep (behavioral, in the child env) ----------------


def _impl(tmp_path, **overrides):
    cfg = dict(
        skill_name="sk",
        permissions=("ws_handler", "companion_process", "supervised_task", "subscribe_event"),
        env_allowlist=(),
        state_dir=tmp_path,
        settings_reader=lambda: {},
        subscribe_events=["skill.lifecycle"],
        companion_processes=[{"name": "daemon", "command": ["python3", "daemon.py"], "runtime": "python3"}],
    )
    cfg.update(overrides)
    return extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(**cfg))


def test_child_rejects_only_unavailable_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_EXTENSION_PROCESS_CHILD", "1")
    assert extension_loader.current_execution_mode() is ExecutionMode.OUT_OF_PROCESS
    api = _impl(tmp_path)

    # Unavailable out-of-process: must raise with companion guidance.
    with pytest.raises(ExtensionRegistrationError, match="register_supervised_task is not available"):
        api.register_supervised_task("bg", lambda: None)
    with pytest.raises(ExtensionRegistrationError, match="subscribe_event is not available"):
        api.subscribe_event("skill.lifecycle", lambda data: None)

    # on_unload is now supported out-of-process (runs at child teardown).
    api.on_unload(lambda: None)
    # send_ws_message is supported (best-effort host relay; no bridge env -> no-op).
    api.send_ws_message("progress", {"pct": 1})


def test_child_companion_is_recorded_for_host_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_EXTENSION_PROCESS_CHILD", "1")
    api = _impl(tmp_path)
    api.register_companion_process("daemon")
    # The child records the manifest-declared name; the host spawns it after
    # catalog. ABI-9: the record lands with the atomic publication the child's
    # load performs after register() returns.
    api._publish_registrations()
    assert "daemon" in extension_loader.list_companion_names()


def test_get_runtime_info_negotiation_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_EXTENSION_PROCESS_CHILD", "1")
    info = _impl(tmp_path).get_runtime_info()
    assert info["execution_mode"] == "out_of_process"
    assert "subscribe_event" not in info["capabilities"]
    assert "on_unload" in info["capabilities"]


# --- M3-lite: durable health vector + regression detection -------------------


def test_health_records_live_then_flags_regression(tmp_path):
    drive = tmp_path
    first = extension_health.record_extension_health(drive, "anime", status="live", version="6.14.0", sha="aaaaaa")
    assert first["regressed"] is False and first["newly_regressed"] is False
    assert first["last_known_good"]["sha"] == "aaaaaa"

    broke = extension_health.record_extension_health(drive, "anime", status="broken", version="6.15.0", sha="bbbbbb", load_error="boom")
    assert broke["regressed"] is True and broke["newly_regressed"] is True
    # last_known_good is preserved from when it was live (for commit-range attribution).
    assert broke["last_known_good"]["sha"] == "aaaaaa"

    again = extension_health.record_extension_health(drive, "anime", status="broken", version="6.15.0", sha="bbbbbb")
    assert again["regressed"] is True and again["newly_regressed"] is False  # not a new transition
    assert (extension_health.read_extension_health(drive, "anime") or {}).get("regressed") is True

    recovered = extension_health.record_extension_health(drive, "anime", status="live", version="6.15.0", sha="cccccc")
    assert recovered["regressed"] is False
    assert extension_health.regressed_extensions(drive) == []


def test_regressed_extensions_skips_uninstalled_or_disabled(tmp_path, monkeypatch):
    extension_health.record_extension_health(tmp_path, "ghost", status="live", sha="a")
    extension_health.record_extension_health(tmp_path, "ghost", status="broken", sha="b")
    # An uninstalled (unresolvable) skill must not raise a permanent false alarm.
    assert extension_health.regressed_extensions(tmp_path) == []
    # The loader's effective switch governs the alarm, including a required package over saved Off.
    loaded = SimpleNamespace(enabled=True)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda dr, name, **k: loaded)
    monkeypatch.setattr("ouroboros.skill_loader.load_enabled", lambda dr, name: False)
    assert [r["skill"] for r in extension_health.regressed_extensions(tmp_path)] == ["ghost"]
    # Installed but effectively disabled -> filtered out.
    loaded.enabled = False
    assert extension_health.regressed_extensions(tmp_path) == []


def test_health_inactive_is_not_a_regression(tmp_path):
    extension_health.record_extension_health(tmp_path, "sk", status="live", sha="a")
    rec = extension_health.record_extension_health(tmp_path, "sk", status="inactive", sha="b")
    assert rec["regressed"] is False
    assert extension_health.regressed_extensions(tmp_path) == []


def test_status_for_runtime_state():
    assert extension_health.status_for_runtime_state({"live_loaded": True}) == "live"
    assert extension_health.status_for_runtime_state(
        {"desired_live": True, "reason": "load_error"}
    ) == "broken"
    assert extension_health.status_for_runtime_state({"desired_live": False, "reason": "disabled"}) == "inactive"


def test_a_per_call_childs_settings_state_is_as_strict_as_the_servers(tmp_path, monkeypatch):
    """The production child reader (``_load_child_extension``) keeps the document's values and the operator's
    overlay, says ``unknown`` for a present document that cannot be read (never ``ok`` with defaults), and an
    opted-in catalog keeps reading its declared snapshot."""
    import json
    import pathlib

    from ouroboros import config as cfg
    from ouroboros import extension_process_runner as runner
    from ouroboros.extension_registry_state import _extensions
    from tests._extension_loader_shared import _prepare_extension

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    key = "TOTAL_BUDGET"
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "oop_state", "def register(api):\n    pass\n", permissions=["read_settings"],
        env_from_settings=[key], extra_frontmatter='plugin_api: "2.0"\n')
    here = pathlib.Path(__file__).resolve().parents[1]

    def child(**kwargs):
        extension_loader.unload_extension(loaded.name)
        runner._load_child_extension(loaded.name, drive_root, here, repo_root, **kwargs)
        return _extensions[loaded.name].api_instances[0]

    cfg.SETTINGS_PATH.write_text(json.dumps({key: 77}), encoding="utf-8")
    api = child()
    assert api.get_settings_state([key]) == {"status": "ok", "values": {key: api.get_settings([key])[key]},
                                            "missing": []}
    assert api.get_settings([key])[key] in (77, 77.0)
    cfg.SETTINGS_PATH.write_text("{\"TOTAL_BUDGET\": ", encoding="utf-8")  # present, torn for good
    assert api.get_settings_state([key]) == {"status": "unknown", "values": {}, "missing": []}
    assert key in api.get_settings([key])  # the ordinary read still falls back to defaults
    overlay = child(operator_settings={key: "55"})
    assert overlay.get_settings_state([key])["status"] == "unknown"  # an overlay never hides an unknown document
    cfg.SETTINGS_PATH.unlink()
    assert overlay.get_settings_state([key]) == {"status": "ok", "values": {key: overlay.get_settings([key])[key]},
                                                "missing": []}
    cfg.SETTINGS_PATH.write_text("not json", encoding="utf-8")
    catalog = child(catalog_settings={key: 12})
    assert catalog.get_settings_state([key]) == {"status": "ok", "values": {key: 12}, "missing": []}
    extension_loader.unload_extension(loaded.name)
