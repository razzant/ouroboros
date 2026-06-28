"""Phase 4 regression tests for ``ouroboros.extension_loader``.

Covers PluginAPI surface: register_tool / register_route /
register_ws_handler / register_ui_tab + permission gating +
namespace enforcement + unload cleanup.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import Any, Dict

import pytest

from ouroboros import extension_loader
from ouroboros.extension_companion import CompanionSupervisor, init_server_process_pid
from ouroboros.extension_reconcile_queue import (
    MAX_ATTEMPTS,
    list_extension_reconcile_requests,
    process_extension_reconcile_requests,
    request_extension_reconcile,
)
from ouroboros.contracts.plugin_api import (
    FORBIDDEN_EXTENSION_SETTINGS,
    PluginAPI,
    VALID_EXTENSION_PERMISSIONS,
)
from ouroboros.skill_loader import (
    SkillReviewState,
    find_skill,
    save_enabled,
    save_review_state,
)


from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clear_loader_state(monkeypatch):
    """Reset the module-level registries between tests."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _write_ext_skill(
    repo_root: pathlib.Path,
    name: str,
    *,
    plugin_body: str,
    permissions: list[str],
    env_from_settings: list[str] | None = None,
    entry: str = "plugin.py",
    extra_frontmatter: str = "",
) -> pathlib.Path:
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    perms_yaml = json.dumps(permissions)
    env_yaml = json.dumps(env_from_settings or [])
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Phase 4 extension.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            f"entry: {entry}\n"
            f"permissions: {perms_yaml}\n"
            f"env_from_settings: {env_yaml}\n"
            f"{extra_frontmatter}"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    entry_path = skill_dir / entry
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(plugin_body, encoding="utf-8")
    return skill_dir


def _prepare_extension(
    tmp_path: pathlib.Path,
    name: str,
    plugin_body: str,
    permissions: list[str],
    env_from_settings: list[str] | None = None,
    extra_frontmatter: str = "",
):
    """Write + enable + PASS-review an extension so the loader accepts it."""
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir(exist_ok=True)
    _write_ext_skill(
        repo_root,
        name,
        plugin_body=plugin_body,
        permissions=permissions,
        env_from_settings=env_from_settings,
        extra_frontmatter=extra_frontmatter,
    )
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, name, True)
    save_review_state(
        drive_root,
        name,
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    # Refetch with fresh state on the loaded struct.
    loaded = find_skill(drive_root, name, repo_path=str(repo_root))
    assert loaded is not None
    return loaded, repo_root, drive_root


def _prepare_companion_extension(tmp_path: pathlib.Path, name: str = "compskill"):
    return _prepare_extension(
        tmp_path,
        name,
        "def register(api):\n    api.register_companion_process('daemon')\n",
        permissions=["companion_process"],
        extra_frontmatter=(
            "companion_processes:\n"
            "  - name: daemon\n"
            "    runtime: python3\n"
            "    command: [\"python3\", \"scripts/daemon.py\"]\n"
        ),
    )


def _mark_isolated_deps_installed(drive_root: pathlib.Path, loaded) -> None:
    from ouroboros.marketplace.install_specs import install_specs_hash
    from ouroboros.marketplace.isolated_deps import FINGERPRINT_FILENAME, isolated_env_dir
    from ouroboros.skill_dependencies import auto_install_specs_for_skill
    from ouroboros.skill_loader import skill_state_dir

    auto_specs = auto_install_specs_for_skill(drive_root, loaded)
    assert auto_specs
    payload = {
        "status": "installed",
        "specs_hash": install_specs_hash(auto_specs),
        "installed": auto_specs,
    }
    state_dir = skill_state_dir(drive_root, loaded.name)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "deps.json").write_text(json.dumps(payload), encoding="utf-8")
    env_dir = isolated_env_dir(loaded.skill_dir)
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / FINGERPRINT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _isolated_site_packages_dir(loaded) -> pathlib.Path:
    return (
        loaded.skill_dir
        / ".ouroboros_env"
        / "python"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _add_fake_native_dep(loaded, package_name: str = "dummy_pkg") -> pathlib.Path:
    site_dir = _isolated_site_packages_dir(loaded)
    pkg_dir = site_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("VALUE = 'isolated-native-risk'\n", encoding="utf-8")
    (site_dir / "fake_native.so").write_bytes(b"not a real shared object; scan marker only")
    return site_dir


def test_worker_reconcile_writes_server_marker_for_enable_and_disable(tmp_path: pathlib.Path) -> None:
    init_server_process_pid(999999)
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)

    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    requests = list_extension_reconcile_requests(drive_root)
    assert [item["skill"] for item in requests] == [loaded.name]

    save_enabled(drive_root, loaded.name, False)
    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_unloaded"
    requests = list_extension_reconcile_requests(drive_root)
    assert {item["skill"] for item in requests} == {loaded.name}
    assert "desired_disabled" in {item["reason"] for item in requests}


def test_server_pickup_spawns_stops_and_redrives_missing_companion(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    init_server_process_pid()
    loaded, repo_root, drive_root = _prepare_companion_extension(tmp_path)

    class FakeSupervisor:
        def __init__(self):
            self.runtimes: Dict[str, Dict[str, Any]] = {}
            self.started: list[str] = []
            self.stopped: list[str] = []

        def start(self, descriptor):
            key = f"{descriptor.skill_name}:{descriptor.name}"
            self.runtimes[key] = {"skill_name": descriptor.skill_name, "name": descriptor.name}
            self.started.append(key)
            return True

        def snapshot(self):
            return dict(self.runtimes)

        def stop(self, skill_name: str, name: str):
            self.stopped.append(f"{skill_name}:{name}")
            self.runtimes.pop(f"{skill_name}:{name}", None)

        def stop_skill(self, skill_name: str):
            self.stopped.append(skill_name)
            self.runtimes = {
                key: value
                for key, value in self.runtimes.items()
                if value.get("skill_name") != skill_name
            }

    fake = FakeSupervisor()
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: fake)

    request_extension_reconcile(drive_root, loaded.name, reason="test")
    processed = process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))

    assert processed[0]["skill"] == loaded.name
    assert fake.started == [f"{loaded.name}:daemon"]
    assert list_extension_reconcile_requests(drive_root) == []

    request_extension_reconcile(drive_root, loaded.name, reason="idempotent")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
    assert fake.started == [f"{loaded.name}:daemon"]

    fake.runtimes.clear()
    state = extension_loader.reconcile_extension(
        loaded.name,
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )
    assert state["action"] == "extension_already_live"
    assert fake.started == [f"{loaded.name}:daemon", f"{loaded.name}:daemon"]

    fake.runtimes.clear()
    request_extension_reconcile(drive_root, loaded.name, reason="redrive")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))
    assert fake.started == [
        f"{loaded.name}:daemon",
        f"{loaded.name}:daemon",
        f"{loaded.name}:daemon",
    ]

    save_enabled(drive_root, loaded.name, False)
    request_extension_reconcile(drive_root, loaded.name, reason="disable")
    process_extension_reconcile_requests(drive_root, lambda: {}, repo_path=str(repo_root))

    assert fake.stopped == [f"{loaded.name}:daemon", loaded.name]
    assert fake.snapshot() == {}
    assert list_extension_reconcile_requests(drive_root) == []


def test_pickup_keeps_newer_marker_written_during_processing(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    request_extension_reconcile(drive_root, "race_skill", reason="old")

    def fake_reconcile(skill_name, drive_root_arg, settings_reader, **kwargs):
        request_extension_reconcile(drive_root_arg, skill_name, reason="newer")
        return {"action": "extension_loaded"}

    monkeypatch.setattr(extension_loader, "reconcile_extension", fake_reconcile)
    monkeypatch.setattr(
        extension_loader,
        "ensure_companions_running",
        lambda *args, **kwargs: {"action": "noop"},
    )

    processed = process_extension_reconcile_requests(drive_root, lambda: {})

    assert processed[0]["marker_removed"] is True
    requests = list_extension_reconcile_requests(drive_root)
    assert len(requests) == 1
    assert requests[0]["reason"] == "newer"


def test_repeatedly_failed_marker_moves_out_of_active_queue(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    request_extension_reconcile(drive_root, "broken_skill", reason="test")

    def fake_reconcile(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(extension_loader, "reconcile_extension", fake_reconcile)

    for _ in range(MAX_ATTEMPTS):
        process_extension_reconcile_requests(drive_root, lambda: {})

    assert list_extension_reconcile_requests(drive_root) == []
    failed = list((drive_root / "state" / "extension_reconcile" / "failed").glob("*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text(encoding="utf-8"))["status"] == "failed"


def test_companion_supervisor_exposes_server_redrive_methods() -> None:
    assert callable(getattr(CompanionSupervisor, "snapshot"))
    assert callable(getattr(CompanionSupervisor, "stop_skill"))


def test_server_lifespan_wires_extension_reconcile_pickup() -> None:
    server_py = pathlib.Path(__file__).resolve().parents[1] / "server.py"
    text = server_py.read_text(encoding="utf-8")

    assert "from ouroboros.extension_reconcile_queue import extension_reconcile_pickup_loop" in text
    assert "extension_reconcile_task = asyncio.create_task" in text
    assert "name=\"extension-reconcile-pickup\"" in text
    assert "extension_reconcile_task.cancel()" in text
    assert "await asyncio.wait_for(extension_reconcile_task, timeout=30)" in text


# ---------------------------------------------------------------------------
# PluginAPI contract shape
# ---------------------------------------------------------------------------


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
    assert "GITHUB_TOKEN" in FORBIDDEN_EXTENSION_SETTINGS
    assert "OUROBOROS_NETWORK_PASSWORD" in FORBIDDEN_EXTENSION_SETTINGS


def test_valid_permissions_is_closed_set():
    for needed in ("tool", "route", "ws_handler", "widget", "read_settings", "net", "fs", "subprocess"):
        assert needed in VALID_EXTENSION_PERMISSIONS


# ---------------------------------------------------------------------------
# Successful load + registration
# ---------------------------------------------------------------------------


def test_load_extension_registers_tool(tmp_path):
    plugin = (
        "def _echo(ctx, message='hi'):\n"
        "    return f'echo: {message}'\n"
        "def register(api):\n"
        "    api.register_tool(\n"
        "        'echo',\n"
        "        _echo,\n"
        "        description='echo',\n"
        "        schema={'type': 'object', 'properties': {'message': {'type': 'string'}}},\n"
        "    )\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "ext1", plugin, permissions=["tool"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool_name = extension_loader.extension_surface_name("ext1", "echo")
    tool = extension_loader.get_tool(tool_name)
    assert tool is not None
    assert tool["name"] == tool_name
    assert callable(tool["handler"])


def test_extension_surface_names_are_provider_safe_without_renaming_skill_identity():
    from ouroboros.skill_loader import _sanitize_skill_name

    dotted = "foo.bar"
    unicode_name = "погода"
    dotted_tool = extension_loader.extension_surface_name(dotted, "fetch")
    unicode_tool = extension_loader.extension_surface_name(unicode_name, "fetch")
    generated_token_twin = "foo_bar_336d1b3d72"

    assert _sanitize_skill_name(dotted) == dotted
    assert _sanitize_skill_name("foo_bar") == "foo_bar"
    assert dotted_tool != extension_loader.extension_surface_name("foo_bar", "fetch")
    assert dotted_tool != extension_loader.extension_surface_name(generated_token_twin, "fetch")
    assert extension_loader.extension_surface_name("foo", "bar_baz") != extension_loader.extension_surface_name("foo_bar", "baz")
    for tool_name in (dotted_tool, unicode_tool):
        assert re.match(r"^[A-Za-z0-9_-]{1,64}$", tool_name)
        assert "." not in tool_name
        assert extension_loader.parse_extension_surface_name(tool_name) is not None


def test_on_unload_callback_runs_during_unload(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "cleanup_ext",
        "import pathlib\n"
        "def register(api):\n"
        "    state_dir = pathlib.Path(api.get_state_dir())\n"
        "    api.on_unload(lambda: (state_dir / 'cleanup.txt').write_text('done', encoding='utf-8'))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("cleanup_ext")

    cleanup_file = drive_root / "state" / "skills" / "cleanup_ext" / "cleanup.txt"
    assert cleanup_file.read_text(encoding="utf-8") == "done"
    assert extension_loader.snapshot()["tools"] == []


def test_on_unload_callback_error_does_not_block_teardown(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "bad_cleanup_ext",
        "def register(api):\n"
        "    api.on_unload(lambda: (_ for _ in ()).throw(RuntimeError('boom')))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("bad_cleanup_ext")

    assert extension_loader.snapshot()["tools"] == []


def test_on_unload_callback_cannot_reregister_surfaces(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "ghost_ext",
        "def register(api):\n"
        "    api.on_unload(lambda: api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={}))\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("ghost_ext")

    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []


def test_on_unload_delayed_callback_cannot_reregister_surfaces(tmp_path):
    import time

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "delayed_ghost_ext",
        "import threading, time\n"
        "def register(api):\n"
        "    def cleanup():\n"
        "        def later():\n"
        "            time.sleep(0.1)\n"
        "            try:\n"
        "                api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={})\n"
        "            except Exception:\n"
        "                pass\n"
        "        threading.Thread(target=later).start()\n"
        "    api.on_unload(cleanup)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    extension_loader.unload_extension("delayed_ghost_ext")
    time.sleep(0.3)

    snap = extension_loader.snapshot()
    assert snap["extensions"] == []
    assert snap["tools"] == []


def test_delayed_post_load_registration_is_rejected(tmp_path):
    import time

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "late_register_ext",
        "import threading, time\n"
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n"
        "    def later():\n"
        "        time.sleep(0.1)\n"
        "        try:\n"
        "            api.register_tool('ghost', lambda **kw: 'boo', description='ghost', schema={})\n"
        "        except Exception:\n"
        "            pass\n"
        "    threading.Thread(target=later).start()\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    time.sleep(0.3)

    snap = extension_loader.snapshot()
    assert snap["tools"] == [extension_loader.extension_surface_name("late_register_ext", "ping")]


def test_reconcile_unload_callbacks_do_not_hold_loader_lock(tmp_path):
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "lock_probe",
        "import pathlib, threading\n"
        "def register(api):\n"
        "    state_dir = pathlib.Path(api.get_state_dir())\n"
        "    def cleanup():\n"
        "        done = state_dir / 'snapshot_done.txt'\n"
        "        def worker():\n"
        "            from ouroboros import extension_loader\n"
        "            extension_loader.snapshot()\n"
        "            done.write_text('done', encoding='utf-8')\n"
        "        thread = threading.Thread(target=worker)\n"
        "        thread.start()\n"
        "        thread.join(timeout=1.0)\n"
        "        if not done.exists():\n"
        "            raise RuntimeError('snapshot blocked by loader lock')\n"
        "    api.on_unload(cleanup)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err

    # Make the extension undesired so reconcile unloads it through the normal path.
    save_enabled(drive_root, "lock_probe", False)
    state = extension_loader.reconcile_extension("lock_probe", drive_root, lambda: {})

    done_file = drive_root / "state" / "skills" / "lock_probe" / "snapshot_done.txt"
    assert done_file.read_text(encoding="utf-8") == "done"
    assert state["action"] == "extension_unloaded"


def test_concurrent_reconcile_converges_to_one_live_extension(tmp_path):
    import threading

    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "race_ext",
        "import time\n"
        "def register(api):\n"
        "    time.sleep(0.05)\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )

    results = []
    repo_path = str(tmp_path / "skills")
    threads = [
        threading.Thread(
            target=lambda: results.append(
                extension_loader.reconcile_extension("race_ext", drive_root, lambda: {}, repo_path=repo_path)
            )
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert len(results) == 2
    assert {r["action"] for r in results} <= {"extension_loaded", "extension_already_live"}
    snap = extension_loader.snapshot()
    assert snap["extensions"] == ["race_ext"]
    assert snap["tools"] == [extension_loader.extension_surface_name("race_ext", "ping")]
    assert extension_loader.runtime_state_for_skill_name("race_ext", drive_root, repo_path=repo_path)["reason"] == "ready"


def test_reconcile_extension_allows_warnings_review(tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "advisory_live",
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    save_review_state(
        drive_root,
        "advisory_live",
        SkillReviewState(status="warnings", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "advisory_live", repo_path=str(repo_root))
    assert loaded is not None

    state = extension_loader.reconcile_extension(
        "advisory_live",
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    assert extension_loader.runtime_state_for_skill_name(
        "advisory_live",
        drive_root,
        repo_path=str(repo_root),
    )["reason"] == "ready"


def test_reconcile_extension_allows_warnings_under_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_loader import find_skill
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "advisory_warnings",
        "def register(api):\n"
        "    api.register_tool('ping', lambda **kw: 'pong', description='ping', schema={})\n",
        permissions=["tool"],
    )
    save_review_state(
        drive_root,
        "advisory_warnings",
        SkillReviewState(status="warnings", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "advisory_warnings", repo_path=str(repo_root))
    assert loaded is not None

    state = extension_loader.reconcile_extension(
        "advisory_warnings",
        drive_root,
        lambda: {},
        repo_path=str(repo_root),
    )

    assert state["action"] == "extension_loaded"
    assert state["reason"] == "ready"


def test_load_extension_permission_gate_tool(tmp_path):
    """Extension without 'tool' permission cannot register a tool."""
    plugin = (
        "def _h(ctx): return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('x', _h, description='', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "nopoerm", plugin, permissions=["route"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "'tool'" in err


def test_load_extension_enforces_review_pass(tmp_path):
    """Unreviewed extension is refused (after being enabled)."""
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = "def register(api): pass\n"
    _write_ext_skill(repo_root, "unreviewed", plugin_body=plugin, permissions=[])
    # Enable to get past the "disabled" gate — we want to exercise the
    # review-status gate specifically.
    save_enabled(drive_root, "unreviewed", True)
    loaded = find_skill(drive_root, "unreviewed", repo_path=str(repo_root))
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "executable review" in err


def test_load_extension_refuses_disabled(tmp_path):
    from ouroboros.skill_loader import find_skill
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = "def register(api): pass\n"
    _write_ext_skill(repo_root, "d1", plugin_body=plugin, permissions=[])
    loaded = find_skill(drive_root, "d1", repo_path=str(repo_root))
    assert loaded is not None
    save_review_state(
        drive_root,
        "d1",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    # NOT enabled.
    loaded = find_skill(drive_root, "d1", repo_path=str(repo_root))
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "disabled" in err


def test_reconcile_extension_stays_loaded_in_light_mode(tmp_path, monkeypatch):
    """v5.1.2 Frame A: ``light`` no longer unloads extensions. The
    ``runtime_mode_light`` reason is gone from
    ``_extension_runtime_state``. Extensions follow the same
    enabled / review / content-hash gates regardless of mode.
    """
    plugin = (
        "def _echo(ctx):\n"
        "    return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "lightstop",
        plugin,
        permissions=["tool"],
    )
    grant_roots = []
    real_grant_status = extension_loader.grant_status_for_skill

    def record_grant_root(root, skill):
        grant_roots.append(pathlib.Path(root))
        return real_grant_status(root, skill)

    monkeypatch.setattr(extension_loader, "grant_status_for_skill", record_grant_root)
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    assert grant_roots and grant_roots[0] == drive_root
    assert "lightstop" in extension_loader.snapshot()["extensions"]

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    state = extension_loader.reconcile_extension(
        "lightstop",
        drive_root,
        lambda: {},
        repo_path=repo_root,
    )
    # The ``runtime_mode_light`` reason was removed in v5.1.2; the
    # extension stays live.
    assert state["reason"] != "runtime_mode_light"
    assert state["action"] != "extension_unloaded"
    assert "lightstop" in extension_loader.snapshot()["extensions"]


def test_reconcile_extension_keeps_live_extension_loaded(tmp_path, monkeypatch):
    plugin = (
        "def _echo(ctx):\n"
        "    return 'ok'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "steady",
        plugin,
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    unload_calls: list[str] = []
    monkeypatch.setattr(extension_loader, "unload_extension", unload_calls.append)

    state = extension_loader.reconcile_extension(
        "steady",
        drive_root,
        lambda: {},
        repo_path=repo_root,
    )
    assert state["reason"] == "ready"
    assert state["action"] == "extension_already_live"
    assert unload_calls == []
    assert "steady" in extension_loader.snapshot()["extensions"]


def test_reconcile_extension_reloads_when_live_code_changes(tmp_path):
    from ouroboros.skill_loader import find_skill

    skill_dir = tmp_path / "skills" / "reloadme"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: reloadme\n"
            "description: Live reload.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "env_from_settings: []\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _echo(ctx):\n"
            "    return 'v1'\n"
            "def register(api):\n"
            "    api.register_tool('echo', _echo, description='echo', schema={})\n"
        ),
        encoding="utf-8",
    )
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    save_enabled(drive_root, "reloadme", True)
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "reloadme",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("reloadme", "echo"))
    assert tool is not None
    assert tool["handler"](None) == "v1"

    (skill_dir / "plugin.py").write_text(
        (
            "def _echo(ctx):\n"
            "    return 'v2'\n"
            "def register(api):\n"
            "    api.register_tool('echo', _echo, description='echo', schema={})\n"
        ),
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, "reloadme", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "reloadme",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )

    state = extension_loader.reconcile_extension(
        "reloadme",
        drive_root,
        lambda: {},
        repo_path=skill_dir.parent,
        retry_load_error=True,
    )
    assert state["action"] == "extension_loaded"
    assert state["live_loaded"] is True
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("reloadme", "echo"))
    assert tool is not None
    assert tool["handler"](None) == "v2"


def test_runtime_state_preserves_matching_load_error(tmp_path):
    plugin = (
        "def _hello(request):\n"
        "    return {'hello': 'world'}\n"
        "def register(api):\n"
        "    api.register_route('/absolute', _hello, methods=('GET',))\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "brokenlive",
        plugin,
        permissions=["route"],
    )
    state = extension_loader.reconcile_extension(
        "brokenlive",
        drive_root,
        lambda: {},
        repo_path=repo_root,
        retry_load_error=True,
    )
    assert state["action"] == "extension_load_error"
    refreshed = extension_loader.runtime_state_for_skill_name(
        "brokenlive",
        drive_root,
        repo_path=repo_root,
    )
    assert refreshed["reason"] == "load_error"
    assert "absolute" in str(refreshed["load_error"])
    assert refreshed["live_loaded"] is False


def test_runtime_state_for_skill_name_reports_missing_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    state = extension_loader.runtime_state_for_skill_name(
        "ghost",
        drive_root,
        repo_path=tmp_path / "skills",
    )
    assert state["desired_live"] is False
    assert state["live_loaded"] is False
    assert state["reason"] == "missing"


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


def test_unload_removes_all_registrations(tmp_path):
    plugin = (
        "def _t(c): return 'x'\n"
        "def _r(req): return {}\n"
        "def _w(p): return {}\n"
        "def register(api):\n"
        "    api.register_tool('t', _t, description='', schema={})\n"
        "    api.register_route('r', _r)\n"
        "    api.register_ws_handler('w', _w)\n"
    )
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "full",
        plugin,
        permissions=["tool", "route", "ws_handler"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    snap = extension_loader.snapshot()
    assert snap["tools"] and snap["routes"] and snap["ws_handlers"]

    extension_loader.unload_extension("full")
    snap = extension_loader.snapshot()
    assert snap["tools"] == []
    assert snap["routes"] == []
    assert snap["ws_handlers"] == []
    assert snap["extensions"] == []


def test_reload_all_called_on_settings_save():
    """Phase 4 regression: ``server.py::api_settings_post`` must
    reconcile the live extension registry when OUROBOROS_SKILLS_REPO_PATH
    changes; otherwise switching repo path leaves stale extensions
    registered from the old path."""
    import ast
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ouroboros"
        / "gateway"
        / "settings.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "api_settings_post"
        ):
            body_text = ast.unparse(node)
            assert "reload_all" in body_text or "_reload_extensions" in body_text, (
                "api_settings_post must call extension_loader.reload_all on "
                "OUROBOROS_SKILLS_REPO_PATH change."
            )
            assert "OUROBOROS_SKILLS_REPO_PATH" in body_text
            assert "OUROBOROS_RUNTIME_MODE" in body_text, (
                "api_settings_post must also reconcile extensions when "
                "runtime mode changes."
            )
            return
    assert False, "api_settings_post function not found in gateway/settings.py"


def test_reload_all_called_from_server_startup():
    """Phase 4 regression: server.py main() must call
    ``extension_loader.reload_all`` during startup so enabled extensions
    survive a restart. Without this, only ``toggle_skill`` could ever
    load a plugin. v6.17 also requires the same reload in spawned workers,
    because extension schemas and dispatch registries are process-local."""
    import ast
    src = (pathlib.Path(__file__).resolve().parent.parent / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            body_text = ast.unparse(node)
            assert "_reload_extensions(" in body_text or "reload_all(" in body_text, (
                "server.py does not wire extension_loader.reload_all into startup — "
                "enabled extensions would not survive a process restart."
            )
            assert "if repo_path" not in body_text, (
                "startup extension reload must run even when only bundled "
                "skills are present."
            )
            assert "pytest_default_real_data_dir" in body_text
            assert "Skipping extension reload_all against real DATA_DIR during pytest" in body_text
            break
    else:
        assert False, "lifespan function not found in server.py"

    worker_src = (pathlib.Path(__file__).resolve().parent.parent / "supervisor" / "workers.py").read_text(encoding="utf-8")
    worker_tree = ast.parse(worker_src)
    for node in ast.walk(worker_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "worker_main":
            body_text = ast.unparse(node)
            assert "_reload_extensions(" in body_text or "reload_all(" in body_text, (
                "supervisor worker_main must reload enabled extension tools before make_agent; "
                "otherwise worker processes expose a smaller tool surface than server schemas."
            )
            assert body_text.index("_reload_extensions") < body_text.index("make_agent"), (
                "worker extension reload must happen before make_agent builds ToolRegistry schemas."
            )
            assert "pytest_default_real_data_dir" in body_text
            return
    assert False, "worker_main function not found in supervisor/workers.py"


def test_reload_all_tears_down_stale_extensions(tmp_path):
    """reload_all must unload extensions that no longer exist on disk."""
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "staleish",
        plugin,
        permissions=["tool"],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None
    assert "staleish" in extension_loader.snapshot()["extensions"]
    # Nuke the skill directory; reload_all should tear it down.
    import shutil
    shutil.rmtree(repo_root / "staleish")
    extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))
    assert "staleish" not in extension_loader.snapshot()["extensions"]


def test_reload_all_continues_after_one_extension_exception(tmp_path, monkeypatch, caplog):
    """A reconcile bug in one extension must not block later extensions."""
    import logging

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    for name in ("a_bad", "z_good"):
        _write_ext_skill(repo_root, name, plugin_body=plugin, permissions=["tool"])
        loaded = find_skill(drive_root, name, repo_path=str(repo_root))
        assert loaded is not None
        save_enabled(drive_root, name, True)
        save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=loaded.content_hash))

    original_reconcile = extension_loader.reconcile_extension

    def flaky_reconcile(skill_name, *args, **kwargs):
        if skill_name == "a_bad":
            raise RuntimeError("boom")
        return original_reconcile(skill_name, *args, **kwargs)

    monkeypatch.setattr(extension_loader, "reconcile_extension", flaky_reconcile)

    with caplog.at_level(logging.ERROR):
        results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert "RuntimeError: boom" in results["a_bad"]
    assert results["z_good"] is None
    assert "z_good" in extension_loader.snapshot()["extensions"]
    assert any("Extension reload failed for a_bad; continuing" in rec.message for rec in caplog.records)


def test_reload_all_logs_per_extension_load_error(tmp_path, caplog):
    import logging

    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_ext_skill(
        repo_root,
        "bad_register",
        plugin_body="def register(api):\n    raise RuntimeError('register failed')\n",
        permissions=[],
    )
    loaded = find_skill(drive_root, "bad_register", repo_path=str(repo_root))
    assert loaded is not None
    save_enabled(drive_root, "bad_register", True)
    save_review_state(drive_root, "bad_register", SkillReviewState(status="pass", content_hash=loaded.content_hash))

    with caplog.at_level(logging.ERROR):
        results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert "register failed" in str(results["bad_register"])
    assert any("Extension reload failed for bad_register" in rec.message for rec in caplog.records)


def test_unload_clears_child_module_cache(tmp_path):
    """Phase 4 round 3 regression: unload must purge EVERY
    ``ouroboros._extensions.<skill>.*`` entry from sys.modules, not
    just the top-level module. Otherwise a helper-file edit sticks to
    the stale cached module on reload."""
    import sys as _sys
    skill_dir = tmp_path / "skills" / "tree_ext"
    (skill_dir).mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: tree_ext\n"
            "description: Multi-file extension.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "env_from_settings: []\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "helper.py").write_text("X = 'v1'\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        (
            "from .helper import X\n"
            "def _t(ctx): return X\n"
            "def register(api):\n"
            "    api.register_tool('echo', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    from ouroboros.skill_loader import find_skill
    save_enabled(drive_root, "tree_ext", True)
    loaded = find_skill(drive_root, "tree_ext", repo_path=str(skill_dir.parent))
    assert loaded is not None
    save_review_state(
        drive_root,
        "tree_ext",
        SkillReviewState(status="pass", content_hash=loaded.content_hash),
    )
    loaded = find_skill(drive_root, "tree_ext", repo_path=str(skill_dir.parent))

    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    # Both the package module and its helper child module must live in
    # sys.modules after import, and BOTH must be purged on unload.
    parent_key = extension_loader._module_key("tree_ext")
    child_key = f"{parent_key}.helper"
    assert parent_key in _sys.modules
    assert child_key in _sys.modules
    extension_loader.unload_extension("tree_ext")
    assert parent_key not in _sys.modules
    assert child_key not in _sys.modules


def test_load_extension_requires_explicit_drive_root(tmp_path):
    loaded, _repo_root, _drive_root = _prepare_extension(
        tmp_path,
        "requires_root",
        "def register(api):\n    pass\n",
        [],
    )

    with pytest.raises(TypeError):
        extension_loader.load_extension(loaded, lambda: {})


def test_clean_extension_runtime_state_unloads_staged_import_root(tmp_path):
    loaded, _repo_root, drive_root = _prepare_extension(
        tmp_path,
        "cleanup_ext",
        "def register(api):\n    pass\n",
        [],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    with extension_loader._lock:
        import_root = pathlib.Path(extension_loader._extensions["cleanup_ext"].import_root)
    assert import_root.exists()

    clean_extension_runtime_state()

    assert not import_root.exists()
    assert "cleanup_ext" not in extension_loader.snapshot()["extensions"]


def test_reload_all_sweeps_stale_extension_imports(tmp_path, monkeypatch):
    import os as _os, time as _time, uuid as _uuid
    _DEAD = 999999  # owner PID treated as dead by the stub below
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "sweep_ext",
        "def register(api):\n    pass\n",
        [],
    )
    imports_dir = drive_root / "state" / "skills" / "sweep_ext" / "__extension_imports"
    # A genuine orphan: owner PID dead AND mtime past the spawn grace (per-PID leaf name).
    stale_root = imports_dir / f"{_DEAD}-{_uuid.uuid4().hex}"
    (stale_root / "skill").mkdir(parents=True)
    _old = _time.time() - 2 * extension_loader._IMPORT_SWEEP_GRACE_SEC
    _os.utime(stale_root, (_old, _old))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert results["sweep_ext"] is None
    assert not stale_root.exists()  # dead-owner + past-grace orphan reaped
    live_roots = list(imports_dir.iterdir())
    assert len(live_roots) == 1
    # The freshly-staged tree is tagged with THIS process's PID.
    assert live_roots[0].name.startswith(f"{_os.getpid()}-")
    assert (live_roots[0] / "skill").exists()


def test_reload_all_preserves_live_import_root_while_sweeping_stale_roots(tmp_path, monkeypatch):
    import os as _os, time as _time, uuid as _uuid
    _DEAD = 999999  # dead owner
    _PEER = 888888  # a DIFFERENT, still-alive worker PID
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path,
        "live_sweep_ext",
        "def register(api):\n    pass\n",
        [],
    )
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    with extension_loader._lock:
        live_root = pathlib.Path(extension_loader._extensions["live_sweep_ext"].import_root)
    imports_dir = drive_root / "state" / "skills" / "live_sweep_ext" / "__extension_imports"
    # A LIVE PEER worker's freshly-staged tree (DIFFERENT, alive PID) must NOT be reaped —
    # the cross-worker race the fix targets; the old single-survivor test could not express
    # it, and the 'skip if pid==mine' miscoding would wrongly delete this.
    peer_root = imports_dir / f"{_PEER}-{_uuid.uuid4().hex}"
    (peer_root / "skill").mkdir(parents=True)
    # A genuine orphan: dead owner + mtime past the spawn grace.
    stale_root = imports_dir / f"{_DEAD}-{_uuid.uuid4().hex}"
    (stale_root / "skill").mkdir(parents=True)
    _old = _time.time() - 2 * extension_loader._IMPORT_SWEEP_GRACE_SEC
    _os.utime(stale_root, (_old, _old))
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    results = extension_loader.reload_all(drive_root, lambda: {}, repo_path=str(repo_root))

    assert results["live_sweep_ext"] is None
    assert live_root.exists()       # this process's live bundle tree kept (keep-set + alive)
    assert peer_root.exists()       # a LIVE peer worker's tree NOT reaped (regression direction)
    assert not stale_root.exists()  # dead + aged orphan reaped
    survivors = set(imports_dir.iterdir())
    assert {live_root, peer_root} <= survivors
    assert stale_root not in survivors


def test_sweep_predicate_keeps_live_peer_and_grace_fresh_reaps_dead_orphan(tmp_path, monkeypatch):
    """Per-PID sweep predicate in isolation (empty keep-set): a live-owner tree and a
    dead-owner-but-within-grace tree survive; a dead-owner past-grace tree and a legacy
    bare-uuid tree are reaped. Pins the MAX_WORKERS>1 cross-worker race fix at the
    predicate level — the 'skip if pid==mine' and dropped-grace miscodings both fail here.
    """
    import os as _os, time as _time, uuid as _uuid
    from ouroboros.skill_loader import skill_state_dir
    _DEAD = 999999
    _PEER = 888888  # different, alive
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: pid != _DEAD)
    drive_root = tmp_path / "drive"
    imports_dir = skill_state_dir(drive_root, "predx") / "__extension_imports"
    imports_dir.mkdir(parents=True)

    def _mk(name, age=0.0):
        d = imports_dir / name
        (d / "skill").mkdir(parents=True)
        if age:
            t = _time.time() - age
            _os.utime(d, (t, t))
        return d

    peer_live = _mk(f"{_PEER}-{_uuid.uuid4().hex}")  # owner alive (different PID)
    dead_fresh = _mk(f"{_DEAD}-{_uuid.uuid4().hex}")  # owner dead, fresh
    dead_old = _mk(f"{_DEAD}-{_uuid.uuid4().hex}", age=2 * extension_loader._IMPORT_SWEEP_GRACE_SEC)
    legacy = _mk(_uuid.uuid4().hex)  # bare-uuid legacy (no parseable owner)
    # An all-digit legacy uuid would int-parse to a huge number; the PID-range guard
    # must treat it as legacy (reaped) and NOT feed it to pid_is_alive (which would
    # OverflowError os.kill and escape the sweep).
    all_digit_legacy = _mk("9" * 32)

    # No bundle registered for "predx" -> keep-set empty -> isolates the new predicate.
    extension_loader._sweep_stale_extension_imports(drive_root, "predx")

    assert peer_live.exists(), "a LIVE peer (different PID) tree must NOT be reaped"
    assert dead_fresh.exists(), "a dead-owner tree within the spawn grace must survive"
    assert not dead_old.exists(), "a dead-owner past-grace orphan must be reaped"
    assert not legacy.exists(), "a legacy bare-uuid tree (no live bundle) is reaped as before"
    assert not all_digit_legacy.exists(), "an all-digit legacy uuid is reaped, not crashed on"


def test_tool_registration_collision_raises(tmp_path):
    """Two plugins registering the same tool namespace collide."""
    plugin_a = (
        "def register(api):\n"
        "    api.register_tool('same', lambda ctx: 'a', description='', schema={})\n"
        "    api.register_tool('same', lambda ctx: 'b', description='', schema={})\n"
    )
    loaded, _, drive_root = _prepare_extension(tmp_path, "collider", plugin_a, permissions=["tool"])
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is not None
    assert "already registered" in err
    # Collision raised mid-registration must tear down the first tool too.
    assert extension_loader.snapshot()["tools"] == []


def test_reconcile_reverts_enabled_on_load_error(tmp_path):
    """Atomic enable: a failed enable-time load reverts enabled.json to False."""
    from ouroboros.skill_loader import load_enabled

    plugin = "def register(api):\n    raise RuntimeError('boom in register')\n"
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, "boomext", plugin, permissions=[])
    assert load_enabled(drive_root, "boomext") is True

    state = extension_loader.reconcile_extension(
        "boomext", drive_root, lambda: {}, repo_path=str(repo_root),
        retry_load_error=True, revert_enabled_on_error=True,
    )
    assert state.get("action") == "extension_load_error"
    assert state.get("reverted_enabled") is True
    assert load_enabled(drive_root, "boomext") is False


def test_reconcile_does_not_revert_when_flag_off(tmp_path):
    """Non-enable reconcile (default flag) must not disable a skill on load error."""
    from ouroboros.skill_loader import load_enabled

    plugin = "def register(api):\n    raise RuntimeError('boom')\n"
    loaded, repo_root, drive_root = _prepare_extension(tmp_path, "boomext2", plugin, permissions=[])
    state = extension_loader.reconcile_extension(
        "boomext2", drive_root, lambda: {}, repo_path=str(repo_root), retry_load_error=True,
    )
    assert state.get("action") == "extension_load_error"
    assert load_enabled(drive_root, "boomext2") is True
