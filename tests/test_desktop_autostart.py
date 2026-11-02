"""Sign-in contract across packaged hosts; OS commands never reach the real host."""
from __future__ import annotations

import configparser
import json
import logging
import os
from pathlib import Path
import plistlib
import re
import shlex
from types import SimpleNamespace

import pytest

from ouroboros import desktop_autostart as startup
from ouroboros.launcher_bootstrap import automatic_launch_allowed, parse_launch_options

SHIPPED_UNIT = (Path(__file__).resolve().parents[1] / "packaging/systemd/ouroboros.service.in").read_text(encoding="utf-8")
# The unit exactly as the 7.2.0 through 7.5.1 deb/rpm packages installed it; a managed update never replaces it.
HISTORICAL_UNIT = """[Unit]
Description=Ouroboros agent runtime
Documentation=https://github.com/razzant/ouroboros

[Service]
Type=simple
# Native packages install the release-reviewed launcher at this fixed path.
# The launcher remains the sole owner of bootstrap, restart, panic, and cleanup.
ExecStart=/opt/ouroboros/Ouroboros

# Stop the complete launcher/server/worker tree started by this unit.
KillMode=control-group
KillSignal=SIGTERM
# SIGTERM is sent immediately. This is only the upper bound systemd waits for
# remaining cgroup processes before escalating to SIGKILL.
TimeoutStopSec=120

# Deliberately no systemd restart policy: the launcher owns its crash fuse and
# treats a panic exit as a complete stop until the owner starts Ouroboros again.

[Install]
WantedBy=default.target
"""


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MANAGED_BY_LAUNCHER", "1")
    monkeypatch.setenv("OUROBOROS_PRESENTATION", "desktop_window")
    monkeypatch.setenv("OUROBOROS_APP_VERSION", "7.2.0")
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.delenv("APPDIR", raising=False)
    monkeypatch.delenv("APPIMAGE_EXTRACT_AND_RUN", raising=False)
    monkeypatch.setattr(startup.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.setattr(startup, "NATIVE_UNIT", tmp_path / "ouroboros.service")
    monkeypatch.setattr(startup.os, "getuid", lambda: 501, raising=False)
    state = SimpleNamespace(calls=[], override=None, unit="disabled")

    def run(argv, **kwargs):
        state.calls.append(argv)
        if argv[0] == "launchctl":
            if argv[1] == "enable":
                state.override = "enabled"
                return ""
            assert argv == ["launchctl", "print-disabled", "gui/501"]
            # Live shape (darwin 25): `=> disabled` / `=> enabled`.
            rows = ['"com.apple.Siri.agent" => disabled']  # a neighbour's override is not ours
            rows += [f'"{startup.LABEL}" => {state.override}'] if state.override else []
            return "disabled services = {\n" + "".join(f"\t\t{row}\n" for row in rows) + "\t}"
        assert argv[0:2] == ["systemctl", "--user"] and argv[-1] == "ouroboros.service"
        if argv[2] == "is-enabled":
            return state.unit
        assert argv[2] in {"enable", "disable"}  # never --now, start, stop or linger
        state.unit = "enabled" if argv[2] == "enable" else "disabled"
        return ""

    monkeypatch.setattr(startup, "_run", run)

    def package(platform, *, native=False, appimage=False):
        monkeypatch.setattr(startup, "sys", SimpleNamespace(platform=platform))
        bundle = (tmp_path / "Ouroboros.app/Contents/Resources" if platform == "darwin"
                  else tmp_path / "Ouroboros/_internal")
        exe = (bundle.parent / "MacOS/Ouroboros" if platform == "darwin"
               else bundle.parent / "Ouroboros")
        bundle.mkdir(parents=True, exist_ok=True)
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.touch()
        monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(bundle))
        if native:
            monkeypatch.setattr(startup, "NATIVE_LAUNCHER", exe)
            startup.NATIVE_UNIT.write_text(SHIPPED_UNIT, encoding="utf-8")
        if appimage:
            exe = tmp_path / "My Ouroboros.AppImage"
            exe.touch()
            monkeypatch.setenv("APPIMAGE", str(exe))
            monkeypatch.setenv("APPDIR", str(bundle.parent))
            monkeypatch.setattr(os.path, "ismount", lambda path: str(path) == str(bundle.parent))  # FUSE-mounted
        return exe

    return package, state


def _exec_argv(path: Path) -> list[str]:
    """Exec= as a sign-in session runs it: Desktop Entry string unescaping, then argument quoting."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(path.read_text(encoding="utf-8"))
    escapes = {"s": " ", "n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
    value = re.sub(r"\\(.)", lambda m: escapes.get(m.group(1), m.group(0)), parser["Desktop Entry"]["Exec"])
    return [arg.replace("%%", "%") for arg in shlex.split(value)]


def test_macos_writes_one_launchagent_and_respects_os_override(host, tmp_path):
    package, os_state = host
    exe = package("darwin")
    path = tmp_path / "Library/LaunchAgents/com.ouroboros.agent.plist"
    assert startup.autostart_status() == {"state": "off"}
    assert not path.exists() and os_state.calls == []  # GET writes nothing
    assert startup.autostart_status(True) == {"state": "on"}
    entry = plistlib.loads(path.read_bytes())
    assert entry == {"Label": "com.ouroboros.agent", "ProgramArguments": [str(exe), "--launch-intent", "automatic"], "RunAtLoad": True}
    assert not (tmp_path / ".config").exists()
    assert os_state.calls[0] == ["launchctl", "enable", "gui/501/com.ouroboros.agent"]
    os_state.override = "disabled"  # System Settings or `launchctl disable`, as current macOS prints it
    assert startup.autostart_status()["state"] == "disabled_by_os"
    assert startup.autostart_status(True)["state"] == "on"
    for printed, expected in (("true", "disabled_by_os"), ("false", "on"), (None, "on")):  # older spelling; no row
        os_state.override = printed
        assert startup.autostart_status()["state"] == expected
    entry["ProgramArguments"] = ["/Applications/Other.app/Contents/MacOS/Ouroboros"]
    path.write_bytes(plistlib.dumps(entry))
    assert startup.autostart_status()["state"] == "other_copy"
    assert startup.autostart_status(False) == {"state": "off"}
    assert not path.exists()


@pytest.mark.parametrize("appimage", [False, True])
def test_linux_portable_registers_only_its_stable_target(host, tmp_path, appimage):
    package, os_state = host
    exe = package("linux", appimage=appimage)
    path = tmp_path / ".config/autostart/ouroboros.desktop"
    assert startup.autostart_status()["state"] == "off"
    assert startup.autostart_status(True)["state"] == "on"
    text = path.read_text(encoding="utf-8")
    command = startup._desktop_command(exe)  # escaped: a raw Windows path never equals the written line
    assert text == f"[Desktop Entry]\nType=Application\nName=Ouroboros\nExec={command}\nTerminal=false\n"
    assert _exec_argv(path) == [str(exe), "--launch-intent", "automatic"]
    assert os_state.calls == [] and not (tmp_path / "Library").exists()
    path.write_text(text + "Hidden=true\n", encoding="utf-8")
    assert startup.autostart_status()["state"] == "disabled_by_os"
    assert startup.autostart_status(True)["state"] == "on"
    path.write_text(text + "X-GNOME-Autostart-enabled=true\n", encoding="utf-8")
    assert startup.autostart_status()["state"] == "on"
    path.write_text(text + "X-GNOME-Autostart-enabled=false\n", encoding="utf-8")  # Cinnamon Startup Applications
    assert startup.autostart_status()["state"] == "disabled_by_os"
    assert startup.autostart_status(True)["state"] == "on"
    assert "X-GNOME-Autostart-enabled" not in path.read_text(encoding="utf-8")
    foreign = text.replace(command, '"/other/Ouroboros" --launch-intent automatic')
    path.write_text(foreign + "Hidden=true\n", encoding="utf-8")
    assert startup.autostart_status()["state"] == "off"  # a hidden entry starts no copy at all
    path.write_text(foreign, encoding="utf-8")
    assert startup.autostart_status()["state"] == "other_copy"
    assert startup.autostart_status(False)["state"] == "off"
    assert not path.exists()


@pytest.mark.parametrize("fuse_less", [None, "APPIMAGE_EXTRACT_AND_RUN", "--appimage-extract-and-run"])
def test_appimage_entry_starts_a_fresh_sign_in_session_in_the_same_mode(host, tmp_path, monkeypatch, fuse_less):
    package, _ = host
    exe = package("linux", appimage=True)
    if fuse_less == "APPIMAGE_EXTRACT_AND_RUN":
        monkeypatch.setenv("APPIMAGE_EXTRACT_AND_RUN", "1")  # README's FUSE-less command
    elif fuse_less:
        monkeypatch.setattr(os.path, "ismount", lambda path: False)  # runtime flag: APPDIR is a plain extracted tree
    assert startup.autostart_status(True)["state"] == "on"
    argv = _exec_argv(tmp_path / ".config/autostart/ouroboros.desktop")
    assert argv[0] == str(exe) and argv[-2:] == ["--launch-intent", "automatic"]
    # The pinned type-2 runtime at sign-in, with none of this process's environment: the FIRST long
    # option selects extract-and-run, and the runtime strips that flag before execing AppRun.
    session_env: dict[str, str] = {}
    first = next((arg[2:] for arg in argv[1:] if arg.startswith("--")), None)
    assert ("APPIMAGE_EXTRACT_AND_RUN" in session_env or first == "appimage-extract-and-run") is bool(fuse_less)
    apprun_argv = [arg for arg in argv[1:] if arg != "--appimage-extract-and-run"]
    assert parse_launch_options(apprun_argv).launch_intent == "automatic"
    assert startup.autostart_status()["state"] == "on"  # read back as this copy's own entry


def test_a_copy_started_from_another_appimage_apps_terminal_registers_itself(host, tmp_path, monkeypatch):
    """The AppImage runtime exports APPIMAGE/APPDIR to every descendant: a pair whose AppDir does not hold this
    bundle belongs to another app and names nothing here."""
    package, _ = host
    exe = package("linux")
    (tmp_path / "Editor.AppImage").touch()
    monkeypatch.setenv("APPIMAGE", str(tmp_path / "Editor.AppImage"))
    monkeypatch.setenv("APPDIR", str(tmp_path / ".mount_Editor"))
    assert startup.launcher_target() == (exe, "")


def test_linux_native_uses_shipped_unit_and_replaces_xdg_registration(host, tmp_path):
    package, os_state = host
    package("linux", native=True)
    path = tmp_path / ".config/autostart/ouroboros.desktop"
    path.parent.mkdir(parents=True)
    path.write_text("[Desktop Entry]\nExec=/old/Ouroboros\n", encoding="utf-8")
    assert startup.autostart_status()["state"] == "other_copy"
    assert startup.autostart_status(True)["state"] == "on"
    assert not path.exists()
    assert ["systemctl", "--user", "enable", "ouroboros.service"] in os_state.calls
    os_state.unit = "masked"
    assert startup.autostart_status()["state"] == "disabled_by_os"
    os_state.unit = "enabled"
    assert startup.autostart_status(False)["state"] == "off"
    assert ["systemctl", "--user", "disable", "ouroboros.service"] in os_state.calls


@pytest.mark.parametrize("turned_off", ["Hidden=true", "X-GNOME-Autostart-enabled=false"])
def test_linux_native_unit_is_not_masked_by_a_turned_off_leftover_entry(host, tmp_path, turned_off):
    package, os_state = host
    package("linux", native=True)
    path = tmp_path / ".config/autostart/ouroboros.desktop"
    path.parent.mkdir(parents=True)
    leftover = '[Desktop Entry]\nType=Application\nExec="/home/me/Ouroboros.AppImage" --launch-intent automatic\n'
    path.write_text(leftover + turned_off + "\n", encoding="utf-8")  # turned off in the desktop's startup settings
    assert startup.autostart_status()["state"] == "off"
    os_state.unit = "enabled"
    assert startup.autostart_status()["state"] == "on"
    path.write_text(leftover, encoding="utf-8")  # a live leftover still starts the other copy
    assert startup.autostart_status()["state"] == "other_copy"


def test_an_older_installed_unit_is_never_enabled_but_still_turns_off(host, tmp_path):
    package, os_state = host
    package("linux", native=True)
    startup.NATIVE_UNIT.write_text(HISTORICAL_UNIT, encoding="utf-8")  # old deb/rpm, current managed code
    command = next(line.partition("=")[2] for line in HISTORICAL_UNIT.splitlines() if line.startswith("ExecStart="))
    assert parse_launch_options(shlex.split(command)[1:]).launch_intent == "owner"  # Panic would not hold it
    for enable in (None, True):
        status = startup.autostart_status(enable)
        assert status["state"] == "unavailable" and status["reason"] == startup.UPDATE_PACKAGE
    assert [call[2] for call in os_state.calls] == ["is-enabled", "is-enabled"]  # read, never enabled
    os_state.unit = "enabled"  # registered earlier by hand: shown as it is, so the owner can turn it off
    assert startup.autostart_status() == {"state": "on"}
    path = tmp_path / ".config/autostart/ouroboros.desktop"  # a live entry of another copy beside it
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[Desktop Entry]\nType=Application\nExec="/home/me/Ouroboros.AppImage" --launch-intent automatic\n', encoding="utf-8")
    assert startup.autostart_status() == {"state": "on"}  # not masked as other_copy: enabling is refused, so off must stay reachable
    assert startup.autostart_status(False) == {"state": "off"}
    assert not path.exists()
    assert ["systemctl", "--user", "disable", "ouroboros.service"] in os_state.calls


def test_linux_portable_disables_existing_native_registration_before_writing(host, tmp_path):
    package, os_state = host
    package("linux", appimage=True)
    startup.NATIVE_UNIT.touch()
    os_state.unit = "enabled"
    assert startup.autostart_status()["state"] == "other_copy"
    assert startup.autostart_status(True)["state"] == "on"
    assert os_state.unit == "disabled"
    assert (tmp_path / ".config/autostart/ouroboros.desktop").exists()


@pytest.mark.parametrize("native", [False, True])
def test_linux_refuses_a_second_registration_when_the_first_cannot_be_removed(host, tmp_path, monkeypatch, native):
    package, os_state = host
    package("linux", native=native, appimage=not native)
    startup.NATIVE_UNIT.touch()
    path = tmp_path / ".config/autostart/ouroboros.desktop"
    if native:
        path.parent.mkdir(parents=True)
        path.mkdir()  # unlink must fail before systemctl enable can run
    else:
        os_state.unit = "enabled"
        monkeypatch.setattr(startup, "_run", lambda argv, **kwargs: "enabled")
    with pytest.raises(OSError):
        startup.autostart_status(True)
    assert ["systemctl", "--user", "enable", "ouroboros.service"] not in os_state.calls
    assert not path.is_file()


def test_linux_browser_fallback_still_configures_the_packaged_host(host, monkeypatch):
    package, _ = host
    package("linux")
    monkeypatch.setenv("OUROBOROS_PRESENTATION", "browser_fallback")
    assert startup.autostart_status(True)["state"] == "on"


@pytest.mark.parametrize(("version", "reason"), [
    ("7.1.9", "newer app build"), ("v7.1.12-rc.3", "newer app build"),
    ("", "could not be confirmed"), ("unknown", "could not be confirmed"), ("7.2", "could not be confirmed"),
])
def test_old_or_unproven_launcher_cannot_register(host, monkeypatch, version, reason):
    package, os_state = host
    package("darwin")
    monkeypatch.setenv("OUROBOROS_APP_VERSION", version)
    for enable in (None, True, False):
        status = startup.autostart_status(enable)
        assert status["state"] == "unavailable" and reason in status["reason"]
    assert os_state.calls == []


@pytest.mark.parametrize("version", ["7.2.0", "7.2.0-rc.1", "v7.10.0-rc.2"])
def test_a_prerelease_suffix_does_not_make_a_new_launcher_old(host, monkeypatch, version):
    package, _ = host
    package("darwin")
    monkeypatch.setenv("OUROBOROS_APP_VERSION", version)
    assert startup.autostart_status() == {"state": "off"}


@pytest.mark.parametrize("kind", ["source_launcher", "source_server", "unstable_macos", "missing_appimage"])
def test_unavailable_hosts_never_register(host, tmp_path, monkeypatch, kind):
    package, os_state = host
    exe = package("linux" if kind == "missing_appimage" else "darwin", appimage=kind == "missing_appimage")
    if kind == "source_launcher":
        monkeypatch.setenv("OUROBOROS_BUNDLE_DIR", str(tmp_path / "source"))
    elif kind == "source_server":
        monkeypatch.delenv("OUROBOROS_MANAGED_BY_LAUNCHER")
    elif kind == "unstable_macos":
        monkeypatch.setattr(startup, "is_unstable_macos_app_path", lambda path: True)
    else:
        exe.unlink()
    result = startup.autostart_status(True)
    assert result["state"] == "unavailable" and os_state.calls == []
    if kind == "unstable_macos":
        assert "Move Ouroboros to Applications" in result["reason"]


def test_runtime_context_reports_host_lifecycle_even_for_a_remote_sender(host, tmp_path, monkeypatch):
    from ouroboros import context

    package, _ = host
    package("linux")
    startup.autostart_status(True)
    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, drive_path=lambda name: tmp_path / name)
    monkeypatch.setattr(context, "_runtime_budget_info", lambda *args: {})
    rendered = context.build_runtime_section(env, {"metadata": {"client_surface": {"channel": "web"}}})
    data = json.JSONDecoder().raw_decode(rendered.split("\n\n", 1)[1])[0]
    assert data["runtime_env"]["autostart"] == "on"
    assert data["runtime_env"]["keep_running_after_close"] is False


def test_a_sign_in_start_keeps_a_saved_pause(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor import worker_chat_lane
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _parked(tmp_path, monkeypatch)
    assert queue.persist_queue_snapshot(reason="sign_out")
    workers.PENDING.clear()
    intent = parse_launch_options(["--launch-intent", "automatic"]).launch_intent
    assert automatic_launch_allowed(intent, tmp_path, logging.getLogger(__name__))
    assert queue.restore_pending_from_snapshot() == 1
    monkeypatch.setattr(worker_chat_lane, "_pool", lambda: workers)
    worker_chat_lane.auto_resume_after_restart()
    assert workers.PENDING[0]["_budget_pause"]["exact_continuation"]
    assert budget_pause.budget_pause_row(tmp_path, "pause-task")["state"] == budget_pause.STATE_PAUSED


def test_native_unit_uses_automatic_intent():
    """The unit new deb/rpm packages ship; the adapter enables it only once installed."""
    assert "ExecStart=/opt/ouroboros/Ouroboros --launch-intent automatic" in SHIPPED_UNIT


def test_a_launchagent_whose_copy_is_gone_follows_this_one_without_enabling_anything(host, tmp_path):
    """The app moved or was replaced: the owner's sign-in choice re-points; an OS switch-off and a live copy stay."""
    package, os_state = host
    exe = package("darwin")
    path = tmp_path / "Library/LaunchAgents/com.ouroboros.agent.plist"
    assert startup.follow_moved_app() is False and not path.exists(), "nothing registered: nothing is created"
    other = tmp_path / "Desktop/Ouroboros.app/Contents/MacOS/Ouroboros"
    other.parent.mkdir(parents=True)
    other.touch()
    path.parent.mkdir(parents=True)
    path.write_bytes(plistlib.dumps({"Label": startup.LABEL, "ProgramArguments": [str(other), *startup.AUTOMATIC_ARGS],
                                     "RunAtLoad": True}))
    os_state.override = "disabled"
    assert startup.follow_moved_app() is False and startup.autostart_status() == {"state": "other_copy"}
    other.unlink()  # that copy went to the Trash
    foreign = {"Label": startup.LABEL, "ProgramArguments": [str(other), "--hand-written"], "RunAtLoad": True}
    path.write_bytes(plistlib.dumps(foreign))
    assert startup.follow_moved_app() is False and plistlib.loads(path.read_bytes()) == foreign, "not this app's shape"
    path.write_bytes(plistlib.dumps({"Label": startup.LABEL, "ProgramArguments": [str(other), *startup.AUTOMATIC_ARGS],
                                     "RunAtLoad": True}))
    assert startup.follow_moved_app() is True
    entry = plistlib.loads(path.read_bytes())
    assert entry == {"Label": startup.LABEL, "ProgramArguments": [str(exe), *startup.AUTOMATIC_ARGS], "RunAtLoad": True}
    assert not [argv for argv in os_state.calls if argv[1] == "enable"], "never launchctl enable"
    assert startup.autostart_status() == {"state": "disabled_by_os"}, "the owner's OS switch-off stays"
    assert startup.follow_moved_app() is False


def test_an_xdg_entry_whose_appimage_is_gone_follows_the_new_file_and_keeps_its_switch(host, tmp_path, monkeypatch):
    package, os_state = host
    exe = package("linux", appimage=True)
    monkeypatch.setenv("APPIMAGE_EXTRACT_AND_RUN", "1")  # this copy runs FUSE-less; the rewritten entry must too
    path = tmp_path / ".config/autostart/ouroboros.desktop"
    path.parent.mkdir(parents=True)
    old = tmp_path / "Ouroboros-7.8.0 (old).AppImage"
    path.write_text(f'[Desktop Entry]\nType=Application\nName=Ouroboros\nExec="{old}" --launch-intent automatic\n'
                    "Terminal=false\nX-GNOME-Autostart-enabled=false\n", encoding="utf-8")
    old.touch()
    assert startup.follow_moved_app() is False, "a copy that still exists stays other_copy"
    old.unlink()
    shaped = path.read_text(encoding="utf-8")
    path.write_text(shaped.replace(" --launch-intent automatic", " --minimized"), encoding="utf-8")
    assert startup.follow_moved_app() is False and "--minimized" in path.read_text(encoding="utf-8"), "a hand-written Exec"
    path.write_text(shaped, encoding="utf-8")
    assert startup.follow_moved_app() is True
    assert _exec_argv(path) == [str(exe), "--appimage-extract-and-run", "--launch-intent", "automatic"]
    assert "X-GNOME-Autostart-enabled=false" in path.read_text(encoding="utf-8")
    assert startup.autostart_status() == {"state": "disabled_by_os"} and os_state.calls == []
