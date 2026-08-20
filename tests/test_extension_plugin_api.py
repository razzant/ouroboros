"""The PluginAPI contract shape and its settings access.

Split out of ``tests/test_extension_loader.py`` when that module was divided by
theme; every moved block is verbatim. Covers the runtime-checkable Protocol
match, runtime info reading the live port file, the settings-section lifecycle,
the forbidden-settings and valid-permissions closed sets, and the dual-track
grant model behind ``get_settings``: core keys blocked without an owner grant,
grants bound to the current content hash, and runtime access closing safely
against in-flight readers and unloads.
"""

from __future__ import annotations

import pathlib

from ouroboros import extension_loader
from ouroboros.contracts.plugin_api import (
    FORBIDDEN_EXTENSION_SETTINGS,
    PluginAPI,
    VALID_EXTENSION_PERMISSIONS,
)

from tests._extension_loader_shared import (
    _prepare_extension,
)
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)


def test_plugin_api_impl_matches_protocol():
    """Runtime-checkable Protocol must structurally accept PluginAPIImpl."""
    impl = extension_loader.PluginAPIImpl(
        skill_name="x",
        permissions=(),
        env_allowlist=(),
        state_dir=pathlib.Path("/tmp"),
        settings_reader=lambda: {},
    )
    assert isinstance(impl, PluginAPI)
    info = impl.get_runtime_info()
    assert info["app_version"]
    assert sorted(info) == [
        "app_version",
        "capabilities",
        "data_dir",
        "execution_mode",
        "runtime_mode",
        "server_port",
        "skill_dir",
        "state_dir",
    ]
    # In-process build sees the full capability set including subscribe_event.
    assert info["execution_mode"] == "in_process"
    assert "subscribe_event" in info["capabilities"]


def test_plugin_api_runtime_info_uses_port_file(tmp_path, monkeypatch):
    """server_port must reflect the actual bound server port written by
    server.py/launcher, not the static AGENT_SERVER_PORT fallback."""
    from ouroboros import config as cfg

    port_file = tmp_path / "state" / "server_port"
    port_file.parent.mkdir()
    port_file.write_text("9012\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "PORT_FILE", port_file)
    impl = extension_loader.PluginAPIImpl(
        skill_name="x",
        permissions=(),
        env_allowlist=(),
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
    )

    assert impl.get_runtime_info()["server_port"] == 9012


def test_register_settings_section_lifecycle(tmp_path):
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "settings_ext",
        plugin_body=(
            "def register(api):\n"
            "    api.register_settings_section('config', 'Config', schema={'components': [\n"
            "        {'type': 'markdown', 'text': 'hello'}\n"
            "    ]})\n"
        ),
        permissions=["widget"],
    )

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root, _force_in_process=True)
    assert err is None, err
    sections = extension_loader.snapshot()["settings_sections"]
    assert len(sections) == 1
    assert sections[0]["skill"] == "settings_ext"
    assert sections[0]["section_id"] == "config"

    extension_loader.unload_extension("settings_ext")
    assert extension_loader.snapshot()["settings_sections"] == []


def test_forbidden_extension_settings_carries_repo_secrets():
    """The forbidden-settings tuple must match the repo-credentials set
    ``skill_exec`` already refuses to forward."""
    assert "OPENROUTER_API_KEY" in FORBIDDEN_EXTENSION_SETTINGS
    assert "MINIMAX_API_KEY" in FORBIDDEN_EXTENSION_SETTINGS
    assert "GITHUB_TOKEN" in FORBIDDEN_EXTENSION_SETTINGS
    assert "OUROBOROS_NETWORK_PASSWORD" in FORBIDDEN_EXTENSION_SETTINGS


def test_valid_permissions_is_closed_set():
    for needed in ("tool", "route", "ws_handler", "widget", "read_settings", "net", "fs", "subprocess"):
        assert needed in VALID_EXTENSION_PERMISSIONS


def test_get_settings_blocks_core_keys_without_grant(tmp_path):
    """An extension that lists a core key in env_from_settings without
    an owner grant fails to load and ``PluginAPIImpl.get_settings``
    silently drops the key — the dual-track grant model deliberately
    keeps the failure mode the same as the script path."""
    plugin = (
        "def register(api):\n"
        "    api.register_tool('n', lambda ctx: 'ok', description='n', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "envtest",
        plugin,
        permissions=["tool", "read_settings"],
        env_from_settings=["OPENROUTER_API_KEY", "TIMEZONE", "MY_OK"],
    )
    settings_snapshot = {
        "OPENROUTER_API_KEY": "sk-leak",
        "TIMEZONE": "UTC",
        "MY_OK": "visible",
        "RANDOM_OTHER": "not-allowed",
    }
    err = extension_loader.load_extension(loaded, lambda: settings_snapshot, drive_root=drive_root)
    assert err is not None
    assert "missing owner grants" in err
    assert "OPENROUTER_API_KEY" in err

    impl = extension_loader.PluginAPIImpl(
        skill_name="envtest",
        permissions=["read_settings"],
        env_allowlist=["OPENROUTER_API_KEY", "TIMEZONE", "MY_OK"],
        state_dir=tmp_path,
        settings_reader=lambda: settings_snapshot,
        granted_keys=[],
    )
    got = impl.get_settings(["OPENROUTER_API_KEY", "TIMEZONE", "MY_OK", "RANDOM_OTHER"])
    assert "OPENROUTER_API_KEY" not in got
    assert got["TIMEZONE"] == "UTC"
    assert got["MY_OK"] == "visible"
    assert "RANDOM_OTHER" not in got
    impl._close_runtime_access()
    assert impl.get_settings(["TIMEZONE", "MY_OK"]) == {}


def test_get_settings_rechecks_runtime_close_after_reader_returns(tmp_path):
    import threading

    reader_started = threading.Event()
    release_reader = threading.Event()

    def settings_reader():
        reader_started.set()
        assert release_reader.wait(1.0)
        return {"MY_OK": "visible"}

    impl = extension_loader.PluginAPIImpl(
        skill_name="settings_race",
        permissions=["read_settings"],
        env_allowlist=["MY_OK"],
        state_dir=tmp_path,
        settings_reader=settings_reader,
    )
    result = []
    thread = threading.Thread(target=lambda: result.append(impl.get_settings(["MY_OK"])))
    thread.start()
    assert reader_started.wait(1.0)
    close_done = threading.Event()
    close_thread = threading.Thread(target=lambda: (impl._close_runtime_access(), close_done.set()))
    close_thread.start()
    assert not close_done.wait(0.1)
    release_reader.set()
    thread.join(timeout=1.0)
    close_thread.join(timeout=1.0)

    assert close_done.is_set()
    assert result == [{}]
    assert impl.get_settings(["MY_OK"]) == {}


def test_unload_does_not_deadlock_with_inflight_get_settings(tmp_path):
    import threading
    import time

    reader_started = threading.Event()
    release_reader = threading.Event()

    def settings_reader():
        reader_started.set()
        release_reader.wait()
        return {"MY_OK": "visible"}

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "settings_unload_race",
        "import threading\n"
        "def register(api):\n"
        "    threading.Thread(target=lambda: api.get_settings(['MY_OK'])).start()\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool", "read_settings"],
        env_from_settings=["MY_OK"],
    )
    err = extension_loader.load_extension(loaded, settings_reader, drive_root=drive_root)
    assert err is None, err
    assert reader_started.wait(1.0)

    unload_done = threading.Event()
    unload_thread = threading.Thread(target=lambda: (extension_loader.unload_extension("settings_unload_race"), unload_done.set()))
    unload_thread.start()
    time.sleep(0.1)
    release_reader.set()
    unload_thread.join(timeout=1.0)

    assert unload_done.is_set()
    assert extension_loader.snapshot()["extensions"] == []


def test_load_extension_rejects_grant_with_stale_content_hash(tmp_path):
    """v5.2.2 dual-track grants: the loader binds the persisted grant
    to the current content hash. A grants.json written for a prior
    revision must NOT authorise the freshly-edited plugin (defense in
    depth — even if ``grant_status_for_skill`` is bypassed)."""
    from ouroboros.skill_loader import save_skill_grants

    plugin = (
        "def register(api):\n"
        "    api.register_tool('n', lambda ctx: 'ok', description='n', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "stale_grant",
        plugin,
        permissions=["tool", "read_settings"],
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    # Persist a grant with the WRONG content hash — simulates a manifest
    # / plugin edit that the operator has not re-authorised.
    save_skill_grants(
        drive_root,
        "stale_grant",
        ["OPENROUTER_API_KEY"],
        content_hash="some-other-hash",
        requested_keys=["OPENROUTER_API_KEY"],
    )
    err = extension_loader.load_extension(
        loaded,
        lambda: {"OPENROUTER_API_KEY": "sk-secret"},
        drive_root=drive_root,
    )
    assert err is not None
    assert "missing owner grants" in err


def test_get_settings_returns_core_key_with_grant(tmp_path):
    """An owner-granted core key is forwarded to the in-process plugin
    via ``PluginAPIImpl.get_settings``. The grant must be bound to the
    current content hash + manifest-requested set; ``load_extension``
    enforces both before constructing the API impl."""
    from ouroboros.skill_loader import save_skill_grants

    plugin = (
        "def register(api):\n"
        "    api.register_tool('n', lambda ctx: 'ok', description='n', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "granted_ext",
        plugin,
        permissions=["tool", "read_settings"],
        env_from_settings=["OPENROUTER_API_KEY", "TIMEZONE"],
    )
    save_skill_grants(
        drive_root,
        "granted_ext",
        ["OPENROUTER_API_KEY"],
        content_hash=loaded.content_hash,
        requested_keys=["OPENROUTER_API_KEY"],
    )
    settings_snapshot = {
        "OPENROUTER_API_KEY": "sk-allowed",
        "TIMEZONE": "UTC",
    }
    err = extension_loader.load_extension(loaded, lambda: settings_snapshot, drive_root=drive_root)
    assert err is None, err

    impl = extension_loader.PluginAPIImpl(
        skill_name="granted_ext",
        permissions=["read_settings"],
        env_allowlist=["OPENROUTER_API_KEY", "TIMEZONE"],
        state_dir=tmp_path,
        settings_reader=lambda: settings_snapshot,
        granted_keys=["OPENROUTER_API_KEY"],
    )
    got = impl.get_settings(["OPENROUTER_API_KEY", "TIMEZONE"])
    assert got.get("OPENROUTER_API_KEY") == "sk-allowed"
    assert got.get("TIMEZONE") == "UTC"

    # Grant on the WRONG content hash must not authorise — the loader
    # builds an empty granted_keys list and drops the value.
    impl_no_grant = extension_loader.PluginAPIImpl(
        skill_name="granted_ext",
        permissions=["read_settings"],
        env_allowlist=["OPENROUTER_API_KEY", "TIMEZONE"],
        state_dir=tmp_path,
        settings_reader=lambda: settings_snapshot,
        granted_keys=[],
    )
    assert "OPENROUTER_API_KEY" not in impl_no_grant.get_settings(["OPENROUTER_API_KEY"])
