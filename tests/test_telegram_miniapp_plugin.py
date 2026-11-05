from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest


SKILL_ROOT = Path(__file__).parents[1] / "skills" / "telegram"
package = types.ModuleType("telegram_native_test")
package.__path__ = [str(SKILL_ROOT)]
sys.modules["telegram_native_test"] = package
lib_package = types.ModuleType("telegram_native_test.lib")
lib_package.__path__ = [str(SKILL_ROOT / "lib")]
sys.modules["telegram_native_test.lib"] = lib_package
SPEC = importlib.util.spec_from_file_location(
    "telegram_native_test.lib.miniapp_registration",
    SKILL_ROOT / "lib" / "miniapp_registration.py",
)
assert SPEC and SPEC.loader
plugin = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plugin)


class FakeAPI:
    def __init__(self, root: Path, *, port: int = 8765) -> None:
        self.data_dir = root / "data"
        self.state_dir = root / "skill-state"
        self.state_dir.mkdir(parents=True)
        (self.state_dir / "settings.json").write_text(
            json.dumps({"TELEGRAM_CHAT_ID": "12345"}),
            encoding="utf-8",
        )
        self.port = port
        self.routes: list[tuple[str, tuple[str, ...], Any]] = []
        self.companions: list[str] = []
        self.sections: list[tuple[str, str, dict[str, Any]]] = []
        self.logs: list[tuple[str, str]] = []

    @staticmethod
    def _assert_tool_name(name: str) -> None:
        candidate = str(name or "").strip()
        if not candidate or len(candidate) > 64 or not candidate.replace("_", "").isalnum():
            raise ValueError(f"tool name must be alnum/underscore only: {candidate!r}")

    def get_runtime_info(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "state_dir": str(self.state_dir),
            "server_port": self.port,
        }

    def get_state_dir(self) -> str:
        return str(self.state_dir)

    def register_route(self, name: str, handler: Any, methods: tuple[str, ...]) -> None:
        self._assert_tool_name(name)
        self.routes.append((name, methods, handler))

    def register_settings_section(self, section_id: str, title: str, schema: dict[str, Any]) -> None:
        self._assert_tool_name(section_id)
        self.sections.append((section_id, title, schema))

    def register_companion_process(self, name: str) -> None:
        self._assert_tool_name(name)
        self.companions.append(name)

    def log(self, level: str, message: str) -> None:
        self.logs.append((level, message))


def test_register_writes_nonsecret_runtime_config_and_companion(tmp_path: Path) -> None:
    api = FakeAPI(tmp_path, port=9012)
    (api.state_dir / "settings.json").write_text(json.dumps({
        "TELEGRAM_CHAT_ID": "12345", "TELEGRAM_PROXY": "socks5://owner:proxy-secret@127.0.0.1:1080",
    }))
    plugin.register(api)
    config_path = api.state_dir / plugin._CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config == {
        "schema": 2,
        "core_port": 9012,
        "owner_chat_id": 12345,
        "button_text": "Ouroboros",
        "tunnel": "cloudflare_quick",
    }
    if sys.platform != "win32":
        assert config_path.stat().st_mode & 0o777 == 0o600
    assert api.companions == ["miniapp_gateway"]
    assert api.routes == []
    assert api.sections == []


def test_bridge_binding_waits_for_private_owner_and_rejects_symlink(tmp_path: Path) -> None:
    api = FakeAPI(tmp_path)
    settings = api.state_dir / "settings.json"
    settings.write_text(json.dumps({"TELEGRAM_CHAT_ID": "-100123"}), encoding="utf-8")
    plugin.register(api)
    config = json.loads((api.state_dir / plugin._CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["owner_chat_id"] == 0

    target = tmp_path / "other.json"
    target.write_text(json.dumps({"TELEGRAM_CHAT_ID": "999"}), encoding="utf-8")
    settings.unlink()
    settings.symlink_to(target)
    with pytest.raises(Exception, match="unsafe"):
        plugin.register(api)


def test_invalid_core_port_fails_before_companion(tmp_path: Path) -> None:
    api = FakeAPI(tmp_path, port=0)
    with pytest.raises(plugin.ConfigurationError, match="port"):
        plugin.register(api)
    assert api.companions == []


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Darwin", "arm64"),
        ("Darwin", "x86_64"),
        ("Linux", "aarch64"),
        ("Windows", "AMD64"),
    ],
)
def test_supported_platform_matrix(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    monkeypatch.setattr(plugin.platform, "system", lambda: system)
    monkeypatch.setattr(plugin.platform, "machine", lambda: machine)
    assert plugin._platform_error() == ""


def test_unsupported_platform_skips_only_miniapp_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plugin.platform, "system", lambda: "Plan9")
    monkeypatch.setattr(plugin.platform, "machine", lambda: "mips")
    api = FakeAPI(tmp_path)
    plugin.register(api)
    assert api.companions == []
    status = plugin._read_status(api)
    assert status["state"] == "unavailable"
    assert status["reason_code"] == "unsupported_platform"


@pytest.mark.parametrize("build", ["win-amd64", "win32", "unknown"])
def test_empty_windows_machine_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: str,
) -> None:
    from telegram_native_test.scripts import platform_support

    monkeypatch.setattr(plugin.platform, "system", lambda: "Windows")
    monkeypatch.setattr(plugin.platform, "machine", lambda: "")
    monkeypatch.setattr(platform_support.sysconfig, "get_platform", lambda: build)
    api = FakeAPI(tmp_path)
    plugin.register(api)
    if build == "win-amd64":
        assert api.companions == ["miniapp_gateway"]
        assert plugin._read_status(api)["state"] == "starting"
    else:
        assert api.companions == []
        status = plugin._read_status(api)
        assert status["state"] == "unavailable"
        assert status["reason_code"] == "unsupported_platform"


def test_real_companion_environment_platform_consumers(tmp_path, monkeypatch, record_property):
    """Run the real interpreter with the core companion's environment construction.

    Windows CPython 3.10/3.11 reproduces the empty-machine defect; 3.12+ may
    discover a machine through WMI. Other hosts exercise their unchanged path.
    No companion, tunnel, download, or Telegram request is started.
    """
    import subprocess
    import textwrap

    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.extension_companion import _companion_base_env, companion_spawn_env
    from ouroboros.tools import skill_exec

    manifest = parse_skill_manifest_text((SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8"))
    spec = next(item for item in manifest.companion_processes if item["name"] == "miniapp_gateway")
    monkeypatch.setattr(skill_exec, "load_settings", lambda: {"TELEGRAM_BOT_TOKEN": "test-bot-token"})
    # Exactly the descriptor env plus supervisor base env used by start().
    env = {**_companion_base_env(), **companion_spawn_env(
        spec, "test-host-token", env_allow=manifest.env_from_settings,
        granted_upper=manifest.env_from_settings, skill=manifest.name,
        skill_dir=SKILL_ROOT, state_dir=tmp_path / "skill-state",
    )}
    assert env["TELEGRAM_BOT_TOKEN"] == "test-bot-token"
    assert not any(key.upper().startswith("PROCESSOR_") for key in env)
    probe = textwrap.dedent("""
        import json, os, platform, sys, sysconfig
        from pathlib import Path

        sys.path.insert(0, sys.argv[1])
        sys.path.insert(0, str(Path.cwd() / "scripts"))
        from tests.test_telegram_miniapp_plugin import FakeAPI, plugin
        import cloudflare_tunnel as cloudflare
        from platform_support import machine_architecture
        from runtime_status import RuntimeStatus

        assert not any(key.upper().startswith("PROCESSOR_") for key in os.environ)
        api = FakeAPI(Path(os.environ["OUROBOROS_SKILL_STATE_DIR"]).parent)
        plugin.register(api)
        registration = plugin._read_status(api)
        try:
            asset = cloudflare._current_asset().platform_id
        except cloudflare.CloudflaredError:
            asset = None
        RuntimeStatus(api.state_dir, cloudflared_version=cloudflare.CLOUDFLARED_VERSION).publish()
        print(json.dumps({
            "system": platform.system(), "raw_machine": platform.machine(),
            "build": sysconfig.get_platform(), "python": sys.version,
            "implementation": sys.implementation.name, "version": list(sys.version_info[:2]),
            "architecture": machine_architecture(), "asset": asset,
            "companions": api.companions, "registration": registration,
            "status": json.loads((api.state_dir / "status.json").read_text(encoding="utf-8")),
        }))
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe, str(SKILL_ROOT.parents[1].resolve())],
        cwd=SKILL_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    record_property("companion_platform", result.stdout.strip())
    print("COMPANION_PLATFORM " + result.stdout.strip())
    raw = observed["raw_machine"].lower()
    if (observed["system"] == "Windows" and observed["implementation"] == "cpython"
            and observed["version"] in ([3, 10], [3, 11])):
        assert raw == "", observed
    expected = raw
    if observed["system"] == "Windows" and not raw:
        expected = {"win-amd64": "amd64", "win-arm64": "arm64", "win32": "x86"}.get(
            observed["build"].lower(), "",
        )
    assert observed["architecture"] == expected, observed
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(expected, expected)
    supported = ((observed["system"] in {"Darwin", "Linux"} and arch in {"amd64", "arm64"})
                 or (observed["system"] == "Windows" and arch == "amd64"))
    if supported:
        assert observed["asset"] == observed["system"].lower() + "-" + arch, observed
        assert observed["companions"] == ["miniapp_gateway"], observed
        assert observed["registration"]["state"] == "starting", observed
    else:
        assert observed["asset"] is None and observed["companions"] == [], observed
        assert observed["registration"]["reason_code"] == "unsupported_platform", observed
    assert observed["status"]["platform"] == observed["system"].lower() + "-" + expected, observed


def test_windows_status_uses_heartbeat_without_destructive_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeAPI(tmp_path)
    plugin.register(api)
    monkeypatch.setattr(plugin.platform, "system", lambda: "Windows")
    monkeypatch.setattr(plugin, "process_alive", lambda _pid: True)
    monkeypatch.setattr(
        plugin.os,
        "kill",
        lambda *_args: pytest.fail("Windows status must never call os.kill(pid, 0)"),
    )
    (api.state_dir / plugin._STATUS_NAME).write_text(
        json.dumps(
            {
                "state": "ready",
                "message": "ok",
                "public_url": "https://abc.trycloudflare.com/",
                "pid": 4242,
                "updated_at_epoch": int(time.time()),
            }
        ),
        encoding="utf-8",
    )
    assert plugin._read_status(api)["state"] == "ready"


def test_status_projection_returns_only_bounded_diagnostics(tmp_path: Path) -> None:
    api = FakeAPI(tmp_path)
    plugin.register(api)
    extra_key = "TELEGRAM_BOT_TOKEN"
    sentinel = "must-not-leak"
    (api.state_dir / plugin._STATUS_NAME).write_text(
        json.dumps(
            {
                "state": "ready",
                "message": "Existing SPA is available",
                "public_url": "https://abc.trycloudflare.com/",
                "cloudflared_version": "2026.7.2",
                "pid": os.getpid(),
                "updated_at_epoch": int(time.time()),
                "reason_code": "healthy",
                "attempt": 0,
                "last_ready_at_epoch": int(time.time()),
                "next_retry_at_epoch": 0,
                extra_key: sentinel,
                "init_data": sentinel,
            }
        ),
        encoding="utf-8",
    )
    body = plugin._read_status(api)
    assert body["state"] == "ready"
    assert body["message"] == "Existing SPA is available"
    assert body["public_url"] == "https://abc.trycloudflare.com/"
    assert body["cloudflared_version"] == "2026.7.2"
    assert body["reason_code"] == "healthy"
    assert sentinel not in json.dumps(body)


def test_status_hides_dead_companion_url(tmp_path: Path) -> None:
    api = FakeAPI(tmp_path)
    plugin.register(api)
    (api.state_dir / plugin._STATUS_NAME).write_text(
        json.dumps(
            {
                "state": "ready",
                "message": "old ready state",
                "public_url": "https://dead.trycloudflare.com/",
                "pid": 999_999_999,
                "updated_at_epoch": int(time.time()),
            }
        ),
        encoding="utf-8",
    )
    body = plugin._read_status(api)
    assert body["state"] == "stale"
    assert body["reason_code"] == "heartbeat_stale"
    assert "public_url" not in body


def test_registration_status_becomes_stale_without_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = int(time.time())
    monkeypatch.setattr(plugin.time, "time", lambda: now)
    api = FakeAPI(tmp_path)
    plugin.register(api)
    starting = plugin._read_status(api)
    assert starting["state"] == "starting"
    assert "cloudflared_version" not in starting
    monkeypatch.setattr(plugin.time, "time", lambda: now + 46)
    stale = plugin._read_status(api)
    assert stale["state"] == "stale"
    assert stale["reason_code"] == "heartbeat_stale"


def test_status_pid1_is_never_a_live_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_alive(1) is now true for a bare container server; the companion
    itself is always the server's child, so a status naming PID 1 stays stale."""
    api = FakeAPI(tmp_path)
    plugin.register(api)
    monkeypatch.setattr(plugin, "process_alive", lambda _pid: True)
    status = {"state": "ready", "message": "ok", "public_url": "https://abc.trycloudflare.com/",
              "pid": 1, "updated_at_epoch": int(time.time())}
    path = api.state_dir / plugin._STATUS_NAME
    path.write_text(json.dumps(status), encoding="utf-8")
    assert plugin._read_status(api)["state"] == "stale"
    plugin._seed_starting_status(api)
    assert json.loads(path.read_text(encoding="utf-8"))["reason_code"] == "registered"
