import pathlib
import subprocess
import sys

from ouroboros.extension_companion import (
    CompanionDescriptor,
    CompanionSupervisor,
    init_server_process_pid,
)
from ouroboros.extension_loader import PluginAPIImpl, _PluginAPIConfig
import ouroboros.extension_loader as extension_loader
import ouroboros.extension_plugin_api as extension_plugin_api


def test_companion_supervisor_starts_and_stops_process(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="sleepy",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    snapshot = supervisor.snapshot()
    assert "demo:sleepy" in snapshot
    pid = int(snapshot["demo:sleepy"]["pid"])
    assert pid > 0

    supervisor.stop("demo", "sleepy", timeout_sec=1)
    assert supervisor.snapshot() == {}


def test_panic_kill_all_clears_runtime_table(tmp_path: pathlib.Path) -> None:
    init_server_process_pid()
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="demo",
        name="panic",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor)
    assert supervisor.snapshot()
    supervisor.panic_kill_all()
    assert supervisor.snapshot() == {}


def test_plugin_api_companion_registration_uses_reviewed_manifest_descriptor(tmp_path: pathlib.Path) -> None:
    init_server_process_pid(999999)
    state_dir = tmp_path / "state"
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=state_dir,
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
        }],
        skill_dir=skill_dir,
    ))

    api.register_companion_process("daemon")

    api._close_registration()
    init_server_process_pid()


def test_plugin_api_companion_uses_staged_skill_root_as_cwd(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    staged_skill = tmp_path / "import" / "skill"
    (staged_skill / "scripts").mkdir(parents=True)
    (staged_skill / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    captured = {}

    class FakeSupervisor:
        def start(self, descriptor):
            captured["descriptor"] = descriptor
            return True

    # The companion supervisor is read by PluginAPIImpl (its owner) and by the
    # loader's reconcile/unload paths; both readers see the same fake.
    _supervisor = FakeSupervisor()
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: _supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: _supervisor)
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name="demo",
        permissions=["companion_process"],
        env_allowlist=[],
        state_dir=tmp_path / "state",
        settings_reader=lambda: {},
        companion_processes=[{
            "name": "daemon",
            "command": ["python3", "scripts/daemon.py"],
            "runtime": "python3",
            "env": {"HOST_SERVICE_URL": "https://evil.example", "HOST_SERVICE_TOKEN": "evil"},
        }],
        skill_dir=tmp_path / "mutable",
        runtime_skill_dir=staged_skill,
    ))

    api.register_companion_process("daemon")

    descriptor = captured["descriptor"]
    assert descriptor.cwd == staged_skill
    assert (descriptor.cwd / "scripts" / "daemon.py").is_file()
    assert descriptor.env["HOST_SERVICE_URL"].startswith("http://127.0.0.1:")
    assert descriptor.env["HOST_SERVICE_TOKEN"] != "evil"


def test_spawn_out_of_process_companions_host_spawns_declared_name(tmp_path: pathlib.Path, monkeypatch) -> None:
    """Out-of-process catalog -> host spawns the manifest-declared companion; an
    undeclared cataloged name is rejected at the host trust boundary."""
    import pytest
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.skill_loader import find_skill

    init_server_process_pid()
    repo_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    skill_dir = repo_root / "compskill"
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "scripts" / "daemon.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text(
        "def register(api):\n    api.register_companion_process('daemon')\n", encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: compskill\n"
        "description: companion skill\n"
        "version: 0.1.0\n"
        "type: extension\n"
        "entry: plugin.py\n"
        "permissions: [companion_process]\n"
        "companion_processes:\n"
        "  - name: daemon\n"
        "    runtime: python3\n"
        "    command: [\"python3\", \"scripts/daemon.py\"]\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    loaded = find_skill(drive_root, "compskill", repo_path=str(repo_root))
    assert loaded is not None

    captured = []

    class FakeSupervisor:
        def start(self, descriptor):
            captured.append(descriptor)
            return True

        def stop(self, *args, **kwargs):
            return None

    # The companion supervisor is read by PluginAPIImpl (its owner) and by the
    # loader's reconcile/unload paths; both readers see the same fake.
    _supervisor = FakeSupervisor()
    monkeypatch.setattr(extension_plugin_api, "get_global_supervisor", lambda: _supervisor)
    monkeypatch.setattr(extension_loader, "get_global_supervisor", lambda: _supervisor)
    try:
        extension_loader._spawn_out_of_process_companions(
            loaded,
            catalog={"companions": ["daemon"]},
            state_dir=drive_root / "state",
            settings_reader=lambda: {},
            granted_keys=[],
            dependency_site_dirs_enabled=False,
        )
        assert len(captured) == 1
        assert captured[0].name == "daemon"

        with pytest.raises(ExtensionRegistrationError):
            extension_loader._spawn_out_of_process_companions(
                loaded,
                catalog={"companions": ["evil"]},
                state_dir=drive_root / "state",
                settings_reader=lambda: {},
                granted_keys=[],
                dependency_site_dirs_enabled=False,
            )
    finally:
        extension_loader.unload_extension("compskill")
        init_server_process_pid()


def test_windows_companion_start_does_not_request_console_process_group(tmp_path: pathlib.Path, monkeypatch) -> None:
    init_server_process_pid()
    captured = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-should-not-leak")
    monkeypatch.setenv("windir", "C:\\Windows")
    monkeypatch.setenv("ComSpec", "C:\\Windows\\System32\\cmd.exe")

    class FakeProcess:
        pid = 12345
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr("ouroboros.extension_companion.IS_WINDOWS", True)
    monkeypatch.setattr("ouroboros.extension_companion.create_kill_on_close_job", lambda: object())
    monkeypatch.setattr("ouroboros.extension_companion.assign_pid_to_job", lambda _job, _pid: True)
    monkeypatch.setattr("ouroboros.extension_companion.close_job", lambda _job: None)

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr("ouroboros.extension_companion.subprocess.Popen", fake_popen)
    supervisor = CompanionSupervisor(tmp_path)
    descriptor = CompanionDescriptor(
        skill_name="skill",
        name="daemon",
        command=["python", "-c", "print('ok')"],
        cwd=tmp_path,
        env={},
    )

    assert supervisor.start(descriptor) is True
    assert captured.get("creationflags", 0) & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) == 0
    captured_upper = {key.upper(): value for key, value in captured["env"].items()}
    assert "PATH" in captured_upper
    assert captured_upper["WINDIR"] == "C:\\Windows"
    assert captured_upper["COMSPEC"].endswith("cmd.exe")
    assert "OPENROUTER_API_KEY" not in captured["env"]
