"""Deployment roots and the actual endpoint, read the same way by every consumer.

The port file a server publishes follows ``OUROBOROS_PORT_FILE`` for the root it hosts,
and only that root: CLI, Settings, in- and out-of-process extensions, the launcher and
the browser's per-root endpoint identity agree, while a foreign root reads its own file.
A direct server names its executing checkout before any import resolves the repository;
the launcher requests the configured port; a fallback never lands on this installation's
service ports; WORLD.md and evolution stats name the roots actually used.
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    alias = tmp_path / "data-alias"
    alias.symlink_to(data, target_is_directory=True)
    relocated = tmp_path / "run" / "ouroboros.port"
    relocated.parent.mkdir()
    relocated.write_text("9311", encoding="utf-8")
    (data / "state" / "server_port").write_text("8765", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_PORT_FILE", str(relocated))
    return data, alias, relocated


def test_the_override_relocates_only_the_hosting_roots_port_file(hosted, tmp_path):
    from ouroboros.config import resolve_port_file

    data, alias, relocated = hosted
    foreign = tmp_path / "task-drive"
    assert resolve_port_file() == relocated
    assert resolve_port_file(alias) == relocated, "the hosting root through another spelling"
    assert resolve_port_file(foreign) == foreign / "state" / "server_port"
    assert resolve_port_file(foreign, hosting_data_dir=foreign) == relocated  # a server names its own root
    os.environ.pop("OUROBOROS_PORT_FILE")
    assert resolve_port_file() == data / "state" / "server_port"


def test_cli_settings_and_endpoint_identity_read_the_actual_port(hosted, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import cli, server_process
    from ouroboros.gateway import settings as gateway_settings

    data, _alias, relocated = hosted
    monkeypatch.delenv("OUROBOROS_URL", raising=False)
    assert cli._default_base_url().endswith(":9311")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(gateway_settings, "DATA_DIR", data)
    assert gateway_settings._port_file(request) == relocated
    monkeypatch.setattr(server_process, "DATA_DIR", data)
    assert server_process._port_file_value(data) == 9311
    foreign = tmp_path / "foreign"
    (foreign / "state").mkdir(parents=True)
    (foreign / "state" / "server_port").write_text("9400", encoding="utf-8")
    assert server_process._port_file_value(foreign) == 9400, "a foreign root never borrows this override"
    # The browser policy's identity reader treats the relocated main port as this root's expected endpoint.
    identity = server_process.runtime_service_identity(data, 9311, lambda _host: True)
    assert identity == server_process.SERVICE_IDENTITY_UNKNOWN
    assert server_process.runtime_service_identity(foreign, 9311, lambda _host: True) == ""


def test_out_of_process_extension_children_carry_the_hosting_port_file(hosted, tmp_path, monkeypatch):
    from ouroboros import extension_process_runner as runner

    data, _alias, relocated = hosted
    env = runner._child_env(drive_root=data, repo_dir=REPO, skill_name="probe", skill_dir=tmp_path,
                            env_allowlist=[], granted_keys=[])
    assert env["OUROBOROS_PORT_FILE"] == str(relocated) and env["OUROBOROS_DATA_DIR"] == str(data)
    monkeypatch.delenv("OUROBOROS_PORT_FILE")
    env = runner._child_env(drive_root=data, repo_dir=REPO, skill_name="probe", skill_dir=tmp_path,
                            env_allowlist=[], granted_keys=[])
    assert "OUROBOROS_PORT_FILE" not in env


_REPO_PROBE = r'''
import json, os, pathlib, sys
import server
from ouroboros import config
from supervisor import git_ops
print(json.dumps({"env": os.environ.get("OUROBOROS_REPO_DIR"), "server": str(server.REPO_DIR),
                  "config": str(config.REPO_DIR), "git_ops": str(git_ops.current_repo_dir()),
                  "port_file": str(server.PORT_FILE),
                  "lock": (pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]) / "state" / "server.lock").exists()}))
'''


@pytest.mark.serial
@pytest.mark.parametrize("explicit", [False, True])
def test_a_direct_server_names_its_executing_checkout_before_imports(tmp_path, explicit):
    data = tmp_path / "data"
    env = {k: v for k, v in os.environ.items() if k not in {"OUROBOROS_REPO_DIR", "OUROBOROS_APP_ROOT"}}
    # A synthetic HOME keeps the pytest child-env guard from re-adding its default repository.
    env.update(OUROBOROS_DATA_DIR=str(data), OUROBOROS_PORT_FILE=str(tmp_path / "port"),
               OUROBOROS_APP_ROOT=str(tmp_path / "unrelated-app-root"), PYTHONPATH=str(REPO),
               HOME=str(tmp_path / "home"))
    if explicit:
        env["OUROBOROS_REPO_DIR"] = str(tmp_path / "explicit-repo")
    run = subprocess.run([sys.executable, "-c", _REPO_PROBE], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    facts = json.loads(run.stdout.strip().splitlines()[-1])
    expected = str(tmp_path / "explicit-repo") if explicit else str(REPO)
    assert facts["env"] == facts["server"] == facts["config"] == facts["git_ops"] == expected
    assert facts["port_file"] == str(tmp_path / "port"), "the server writes the configured port file"
    assert facts["lock"] is False, "importing the server admits nothing"


def test_the_launcher_requests_the_configured_port_and_stamps_its_child(monkeypatch, tmp_path):
    import launcher

    monkeypatch.delenv("OUROBOROS_SERVER_PORT", raising=False)
    assert launcher._requested_server_port() == launcher.AGENT_SERVER_PORT
    monkeypatch.setenv("OUROBOROS_SERVER_PORT", "9234")
    assert launcher._requested_server_port() == 9234

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "server.py").write_text("print('server')\n", encoding="utf-8")
    captured = {}

    class FakeProcess:
        pid = 4243
        stdout = type("Out", (), {"readline": lambda self, _limit=-1: b""})()

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs, cmd=cmd)
        return FakeProcess()

    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(launcher, "REPO_DIR", repo_dir)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", sys.executable)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {})
    monkeypatch.setattr(launcher, "_apply_settings_to_env", lambda _settings: None)
    monkeypatch.setattr(launcher, "_hidden_popen", fake_popen)
    launcher.start_agent(port=launcher._requested_server_port())
    assert captured["env"]["OUROBOROS_SERVER_PORT"] == "9234"
    assert captured["env"]["OUROBOROS_LAUNCHER_STOP_STDIN"] == "1" and captured["stdin"] is subprocess.PIPE
    record = json.loads((tmp_path / "data" / "state" / "server_process.json").read_text(encoding="utf-8"))
    assert record["port"] == 9234 and isinstance(record.get("fingerprint"), dict)


def _consecutive_free_ports(count: int) -> int:
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base + count >= 65535:
            continue
        sockets = []
        try:
            for port in range(base, base + count):
                sock = socket.socket()
                sock.bind(("127.0.0.1", port))
                sockets.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    pytest.skip("no run of consecutive free loopback ports")


@pytest.mark.serial
def test_a_fallback_port_skips_this_installations_service_ports():
    from ouroboros.server_entrypoint import find_free_port

    base = _consecutive_free_ports(5)
    busy = [socket.socket(), socket.socket()]
    try:
        for sock, port in zip(busy, (base, base + 1)):
            sock.bind(("127.0.0.1", port))
            sock.listen()
        assert find_free_port("127.0.0.1", base, wait_retries=1) == base + 2
        assert find_free_port("127.0.0.1", base, wait_retries=1, reserved={base + 2, base + 3}) == base + 4
    finally:
        for sock in busy:
            sock.close()


def test_reserved_service_ports_follow_settings_and_environment(monkeypatch):
    from ouroboros import server_entrypoint

    monkeypatch.setattr("ouroboros.config.load_settings",
                        lambda: {"OUROBOROS_HOST_SERVICE_PORT": "9767", "LOCAL_MODEL_PORT": "bogus"})
    assert server_entrypoint.reserved_service_ports() == {9767, 8766}


def test_world_profile_names_the_roots_in_use_even_with_a_quote_in_the_path(tmp_path, monkeypatch):
    """The launcher's own command (paths ride argv) generates WORLD.md for an apostrophe path."""
    import logging

    from ouroboros import launcher_bootstrap

    data = tmp_path / "Bob's data root"
    context = launcher_bootstrap.BootstrapContext(
        bundle_dir=tmp_path, repo_dir=REPO, data_dir=data, settings_path=data / "settings.json",
        embedded_python=sys.executable, app_version="0", hidden_run=subprocess.run,
        save_settings=lambda _settings: None, log=logging.getLogger("world-profile-test"))
    monkeypatch.setattr(launcher_bootstrap, "ensure_managed_repo", lambda _context: "unchanged")
    assert launcher_bootstrap.bootstrap_repo(context) is True
    world = (data / "memory" / "WORLD.md").read_text(encoding="utf-8")
    assert f"`{REPO}` contains my codebase" in world and f"`{data}` contains my memory" in world
    assert "~/Ouroboros" not in world


def test_the_world_profile_command_suits_a_released_seeds_one_argument_profiler(tmp_path, monkeypatch):
    """A new launcher over the v7.6.0 seed body still generates WORLD.md with that body's profiler."""
    import io
    import logging
    import tarfile

    from ouroboros import launcher_bootstrap

    archive = subprocess.run(["git", "archive", "v7.6.0", "ouroboros"], cwd=str(REPO), capture_output=True)
    if archive.returncode != 0:
        pytest.skip("the v7.6.0 release tag is not available in this clone")
    seed_body = tmp_path / "seed body"
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(seed_body)  # noqa: S202 - this repository's own release tree
    data = tmp_path / "data"
    context = launcher_bootstrap.BootstrapContext(
        bundle_dir=tmp_path, repo_dir=seed_body, data_dir=data, settings_path=data / "settings.json",
        embedded_python=sys.executable, app_version="7.6.0", hidden_run=subprocess.run,
        save_settings=lambda _settings: None, log=logging.getLogger("world-profile-test"))
    monkeypatch.setattr(launcher_bootstrap, "ensure_managed_repo", lambda _context: "unchanged")
    assert launcher_bootstrap.bootstrap_repo(context) is True
    assert (data / "memory" / "WORLD.md").read_text(encoding="utf-8").startswith("# WORLD.md")


def test_evolution_stats_reads_the_serving_repository_per_call(monkeypatch, tmp_path):
    from ouroboros.tools import evolution_stats

    seen = []
    monkeypatch.setattr(evolution_stats.subprocess, "run",
                        lambda cmd, **kw: seen.append(kw["cwd"]) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(tmp_path / "serving"))
    evolution_stats._git(["status"])
    assert seen == [str(tmp_path / "serving")]
