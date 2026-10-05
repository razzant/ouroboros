"""Bound host values and their launch source survive ordinary Settings/restart flows."""

import json
import logging
import os
import pathlib
import sys

import pytest

from ouroboros import config, local_model, platform_layer, server_process
from ouroboros.gateway import settings
from ouroboros.server_control import restart_current_process
from ouroboros.server_entrypoint import parse_server_args
from tests.test_server_shutdown import _stop_restart_watcher


@pytest.fixture
def applied_host(monkeypatch, tmp_path):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    monkeypatch.setattr(server_process, "_applied_server_host_source", "unknown")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(local_model, "_manager", local_model.LocalModelManager())
    saved = dict(config.SETTINGS_DEFAULTS)
    server_process.record_applied_restart_settings({key: saved[key] for key in (
        "OUROBOROS_MAX_WORKERS", "OUROBOROS_SERVER_HOST", "OUROBOROS_HOST_SERVICE_PORT",
        "OUROBOROS_SKILLS_REPO_PATH",
    )}, server_host_source="settings")
    return saved


@pytest.mark.parametrize("source", ["settings", "environment", "cli", "launcher"])
def test_known_bind_is_separate_from_host_source(applied_host, tmp_path, source):
    saved = {**applied_host, "OUROBOROS_SERVER_HOST": "0.0.0.0"}
    server_process.record_applied_restart_settings({"OUROBOROS_SERVER_HOST": "127.0.0.1"},
                                                  server_host_source=source)
    state = settings._build_restart_state(saved)
    assert state["unknown_keys"] == []
    assert state["restart_required"] is (source == "settings")
    assert state["restart_source_unknown_keys"] == (
        ["OUROBOROS_SERVER_HOST"] if source == "launcher" else [])
    if source in {"environment", "cli"}:
        assert "launch configuration overrides" in state["summary"]
        assert "to apply" not in state["summary"]
    if source == "launcher":
        assert "Restart may apply" in state["summary"]
    saved["OUROBOROS_MAX_WORKERS"] += 1
    assert "OUROBOROS_MAX_WORKERS" in settings._build_restart_state(saved)["restart_keys"]
    assert settings._build_restart_state(applied_host)["summary"] == ""


def test_late_launcher_record_resolves_only_matching_process_and_path(applied_host, tmp_path):
    server_process.record_applied_restart_settings({"OUROBOROS_SERVER_HOST": "127.0.0.1"},
                                                  server_host_source="launcher")
    saved = {**applied_host, "OUROBOROS_SERVER_HOST": "0.0.0.0"}
    record_path = tmp_path / "state" / "server_process.json"
    record_path.parent.mkdir()
    expected = {"pid": os.getpid(), "server_path": str(pathlib.Path(server_process.__file__).resolve().parents[1] / "server.py")}
    # An old launcher's equal value never proves where the host came from.
    assert settings._build_restart_state(applied_host)["summary"] == ""
    for record in ({}, expected, {**expected, "server_host_source": {}},
                   {**expected, "pid": os.getpid() + 1, "server_host_source": "settings"},
                   {**expected, "server_path": "/another/server.py", "server_host_source": "settings"}):
        record_path.write_text(json.dumps(record), encoding="utf-8")
        assert settings._build_restart_state(saved)["restart_source_unknown_keys"] == ["OUROBOROS_SERVER_HOST"]
    record_path.write_text("{bad json", encoding="utf-8")
    assert server_process.applied_server_host_source(tmp_path) == "unknown"
    for source in ("settings", "environment"):
        record_path.write_text(json.dumps({**expected, "server_host_source": source}), encoding="utf-8")
        state = settings._build_restart_state(saved)
        assert state["restart_required"] is (source == "settings")
        assert state["restart_source_unknown_keys"] == []
        assert server_process.applied_restart_settings()["OUROBOROS_SERVER_HOST"] == "127.0.0.1"


@pytest.mark.parametrize("argv,env_host,managed,expected_source,expected_host", [
    (["server.py"], None, False, "settings", "127.0.0.1"),
    (["server.py"], "", True, "settings", "127.0.0.1"),
    (["server.py"], "0.0.0.0", False, "environment", "0.0.0.0"),  # Docker/direct env
    (["server.py"], "127.0.0.1", True, "launcher", "127.0.0.1"),
    (["server.py", "--host", "127.0.0.1"], "0.0.0.0", True, "cli", "127.0.0.1"),
])
def test_actual_main_captures_source_after_binding(applied_host, tmp_path, monkeypatch,
                                                  argv, env_host, managed, expected_source, expected_host):
    import server

    class FakeServer:
        def __init__(self, _config):
            self.should_exit = False
            self.stop_watcher_bound = False

        def watch_launcher_stop(self):
            self.stop_watcher_bound = True

        def run(self, *, sockets):
            assert self.stop_watcher_bound
            assert sockets[0].getsockname()[1] > 0
            assert server_process._applied_server_host_source == expected_source
            assert server_process.applied_restart_settings()["OUROBOROS_SERVER_HOST"] == expected_host

    if env_host is None:
        monkeypatch.delenv("OUROBOROS_SERVER_HOST", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_SERVER_HOST", env_host)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(server, "load_settings", lambda: applied_host)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "DEFAULT_PORT", 0)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", managed)
    monkeypatch.setattr(server, "_ACTUAL_BOUND_PORT", None)
    monkeypatch.setattr(server, "get_network_auth_startup_warning", lambda _host: "")
    monkeypatch.setattr(server, "validate_network_auth_configuration", lambda _host: "")
    monkeypatch.setattr(server, "_event_loop", None)
    monkeypatch.setattr(server, "_SignalStopServer", FakeServer)  # the main() server seam (#1142)
    server._restart_requested.clear()
    try:
        assert server.main() == 0
    finally:
        _stop_restart_watcher(server)


@pytest.mark.parametrize("original_host", [None, "", "0.0.0.0"])
@pytest.mark.parametrize("explicit_cli", [False, True])
def test_two_settings_restarts_do_not_invent_environment_authority(monkeypatch, tmp_path, original_host, explicit_cli):
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)  # Exercise the mocked exec seam on every host.
    saved = {"OUROBOROS_SERVER_HOST": "127.0.0.2"}
    monkeypatch.setattr(config, "load_settings", lambda: saved)
    if original_host is None:
        monkeypatch.delenv("OUROBOROS_SERVER_HOST", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_SERVER_HOST", original_host)
    monkeypatch.delenv("OUROBOROS_SERVER_REEXEC_ARGV_JSON", raising=False)
    argv = ["server.py", "--host", "127.0.0.3"] if explicit_cli else ["server.py"]
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(os, "execvpe", lambda executable, argv, env: calls.append((executable, argv, env)))
    actual = "127.0.0.1"
    for desired in ("127.0.0.2", "127.0.0.4"):
        saved["OUROBOROS_SERVER_HOST"] = desired
        restart_current_process(actual, 8765, repo_dir=tmp_path, log=logging.getLogger("test"))
        executable, reexec_argv, env = calls[-1]
        assert reexec_argv == [sys.executable, *argv]
        assert env.get("OUROBOROS_SERVER_HOST") == original_host
        # Simulate the child's actual argument/default selection with the exec boundary held.
        monkeypatch.setattr(sys, "argv", reexec_argv[1:])
        selected = parse_server_args(str(env.get("OUROBOROS_SERVER_HOST") or "").strip() or saved["OUROBOROS_SERVER_HOST"], 8765)
        actual = selected.host
        assert actual == ("127.0.0.3" if explicit_cli else original_host or desired)


def test_public_cli_reexec_argv_and_environment_override_remain_intact(monkeypatch, tmp_path):
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)  # Exercise the mocked exec seam on every host.
    argv = ["-m", "ouroboros", "server", "--host", "0.0.0.0"]
    monkeypatch.setenv("OUROBOROS_SERVER_REEXEC_ARGV_JSON", json.dumps(argv))
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", "0.0.0.0")
    captured = []
    monkeypatch.setattr(os, "execvpe", lambda executable, argv, env: captured.append((argv, env)))
    restart_current_process("0.0.0.0", 8765, repo_dir=tmp_path, log=logging.getLogger("test"))
    assert captured[0][0] == [sys.executable, *argv]
    assert captured[0][1]["OUROBOROS_SERVER_HOST"] == "0.0.0.0"
