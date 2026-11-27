"""Exercise the packaged-desktop seam without loading WinForms or touching the OS shell."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import launcher_windows_runtime as windows

ROOT = Path(__file__).resolve().parents[1]


def lay_install(root, *, registered=True):
    """A per-user install folder; ``registered`` adds Setup's uninstaller and its HKCU values for it."""
    (root / "_internal" / "assets").mkdir(parents=True, exist_ok=True)
    (root / "Ouroboros.exe").touch()
    (root / "_internal" / "assets" / "icon.ico").touch()
    if not registered:
        return {}
    (root / "unins000.exe").touch()
    (root / "unins000.dat").touch()
    return {"Inno Setup: App Path": str(root), "InstallLocation": str(root) + os.sep,  # Setup writes a trailing separator
            "UninstallString": f'"{root / "unins000.exe"}"'}


@pytest.mark.parametrize(("copy", "frozen", "app_ids", "override"), [
    ("installed", True, ["com.ouroboros.agent"], False),
    ("installed", False, ["com.ouroboros.agent"], True),
    ("unregistered_zip_at_install_path", True, [], False),
    ("portable", True, [], False),
    ("portable", False, [], True),
])
def test_distribution_matches_shortcut_without_stealing_portable_group(
    tmp_path, monkeypatch, copy, frozen, app_ids, override,
):
    canonical = tmp_path / "Local AppData" / "Programs" / "Ouroboros"
    distribution = tmp_path / "Portable Ouroboros" if copy == "portable" else canonical
    # Setup's registration names the canonical folder even when a portable copy runs elsewhere; a ZIP unpacked
    # at the canonical path itself has no registration, so the path alone never makes it the installed copy.
    registration = lay_install(canonical, registered=copy != "unregistered_zip_at_install_path")
    lay_install(distribution, registered=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local AppData"))
    monkeypatch.setattr(windows, "IS_WINDOWS", True)
    monkeypatch.setattr(windows, "_setup_registration", lambda: registration)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    assigned = []
    monkeypatch.setattr(windows, "_set_process_app_id", assigned.append)

    seed = distribution / "_internal"
    icon = windows.prepare_windows_desktop_identity(seed)
    assert assigned == app_ids
    assert icon == (seed / "assets" / "icon.ico" if override else None)


@pytest.mark.parametrize("change", [
    None, "no_registration", "other_folder", "missing_dat", "uninstaller_elsewhere", "not_an_uninstaller",
    "install_location_only",
])
def test_setup_ownership_needs_the_registration_naming_this_folder_and_its_uninstaller(tmp_path, monkeypatch, change):
    root = tmp_path / "Programs" / "Ouroboros"
    registration = lay_install(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "unins000.exe").touch()
    (elsewhere / "unins000.dat").touch()
    if change == "no_registration":
        registration = {}
    elif change == "other_folder":
        registration["Inno Setup: App Path"] = str(elsewhere)
    elif change == "missing_dat":
        (root / "unins000.dat").unlink()
    elif change == "uninstaller_elsewhere":
        registration["UninstallString"] = f'"{elsewhere / "unins000.exe"}"'
    elif change == "not_an_uninstaller":
        registration["UninstallString"] = f'"{root / "Ouroboros.exe"}"'
    elif change == "install_location_only":
        del registration["Inno Setup: App Path"]  # then InstallLocation, as IsOwnedInstallDir reads it
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(windows, "IS_WINDOWS", True)
    monkeypatch.setattr(windows, "_setup_registration", lambda: registration)
    owned = change in (None, "install_location_only")
    assert windows.setup_install_root(root / "_internal") == (root if owned else None)
    assert windows.windows_source_host(root / "_internal") is None, "a seed without the packaged leaf hosts nothing"
    leaf = root / "_internal" / "ouroboros" / "launcher_windows_runtime.py"
    leaf.parent.mkdir()
    leaf.touch()
    assert windows.windows_source_host(root / "_internal") == (leaf if owned else None)


def test_missing_icon_and_refused_app_id_do_not_prevent_startup(tmp_path, monkeypatch, caplog):
    distribution = tmp_path / "Programs" / "Ouroboros"
    seed = distribution / "_internal"
    registration = lay_install(distribution)
    (seed / "assets" / "icon.ico").unlink()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(windows, "IS_WINDOWS", True)
    monkeypatch.setattr(windows, "_setup_registration", lambda: registration)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    def refuse(_app_id):
        raise OSError("shell refused")

    monkeypatch.setattr(windows, "_set_process_app_id", refuse)
    assert windows.prepare_windows_desktop_identity(seed) is None
    assert "startup continues" in caplog.text


def test_other_platform_never_asks_for_identity_or_accesses_seed(monkeypatch):
    monkeypatch.setattr(windows, "IS_WINDOWS", False)
    assert windows.prepare_windows_desktop_identity(None) is None
    windows.attach_windows_window_icon(None, None)


def test_the_installer_and_the_app_read_one_registration():
    iss = (ROOT / "packaging" / "windows" / "ouroboros.iss").read_text(encoding="utf-8")
    assert f"UninstallRegKey = '{windows._SETUP_REGISTRATION}';" in iss


@pytest.mark.parametrize(("where", "opened", "error", "args", "console", "refused"), [
    ("elsewhere", 7, 0, [], False, False),  # a portable copy never asks
    ("installed", None, 2, [], False, False),  # no such mutex: nothing changes the installation
    ("installed", 7, 0, [], False, True),  # ordinary Start click exits immediately with no dialog
    ("installed", 7, 0, ["--launch-intent", "automatic"], False, True),  # a sign-in start leaves quietly too
    ("installed", 7, 0, [], True, True),  # the CLI says why on stderr
    ("installed", None, 5, [], False, True),  # any other failure to look refuses too, still without a dialog
])
def test_an_entry_inside_the_installation_refuses_while_setup_holds_its_mutex(
        tmp_path, monkeypatch, capsys, where, opened, error, args, console, refused):
    import ctypes

    root = tmp_path / "Programs" / "Ouroboros"
    exe = (root if where == "installed" else tmp_path / "Tools" / "Ouroboros") / "Ouroboros.exe"
    asked, messages = [], []
    kernel = SimpleNamespace(OpenMutexW=lambda access, inherit, name: asked.append(name) or opened, CloseHandle=asked.append)
    monkeypatch.setattr(windows, "IS_WINDOWS", True)
    monkeypatch.setattr(windows, "_setup_root", lambda: str(root))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    monkeypatch.setattr(windows, "_show_windows_message", lambda title, message: messages.append(message))
    if refused:
        with pytest.raises(SystemExit) as stop:
            windows.refuse_while_setup_changes(str(exe), args, console=console)
        assert stop.value.code == 75
    else:
        windows.refuse_while_setup_changes(str(exe), args, console=console)
    assert asked[:1] == ([] if where == "elsewhere" else [windows._SETUP_CHANGING])
    assert messages == [] and ("changing this installation" in capsys.readouterr().err) is (refused and console)


@pytest.mark.parametrize(("platform", "frozen", "asks"), [("win32", True, True), ("win32", False, False),
                                                           ("darwin", True, False)])
def test_ouroboros_exe_asks_before_any_package_work_and_a_checkout_launcher_never_does(monkeypatch, platform, frozen,
                                                                                       asks):
    """Run as the entry script, launcher.py asks first as the frozen Windows exe; any other start goes on without
    asking, to the next package import (ouroboros.config, made to fail here, which ends the run)."""
    import runpy

    import launcher  # noqa: F401  (its stdlib imports load before the fake platform below)

    asked = []

    def refuse(executable, args=()):
        asked.append((executable, args))
        raise SystemExit(75)

    monkeypatch.setattr(windows, "refuse_while_setup_changes", refuse)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(sys, "argv", ["Ouroboros.exe", "--launch-intent", "automatic"])
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)  # the script sets both, as it must
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setitem(sys.modules, "ouroboros.config", None)
    with pytest.raises(SystemExit if asks else ImportError) as stop:
        runpy.run_path(str(ROOT / "launcher.py"), run_name="__main__")
    assert asked == ([(sys.executable, ["--launch-intent", "automatic"])] if asks else [])
    if asks:
        assert stop.value.code == 75
    else:
        assert "ouroboros.config" in str(stop.value)


def test_the_shipped_cli_refuses_in_its_own_process_before_its_package_works(monkeypatch):
    """bin\\ouroboros.cmd runs ``python -m ouroboros.packaged_cli``: the refusal comes before its next package import."""
    import runpy

    asked = []

    def refuse(executable, args=(), *, console=False):
        asked.append((executable, console))
        raise SystemExit(75)

    monkeypatch.setattr(windows, "refuse_while_setup_changes", refuse)
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)  # the module sets both, as it must
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delitem(sys.modules, "ouroboros.git_selection", raising=False)
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("ouroboros.packaged_cli", run_name="__main__")
    assert stop.value.code == 75 and asked == [(sys.executable, True)]
    assert "ouroboros.git_selection" not in sys.modules


@pytest.mark.parametrize("flags", [["-E", "-s", "-B"], ["-I", "-B"]])  # with and without a script folder
def test_the_handoff_entry_runs_the_checkout_launcher_as_python_would(tmp_path, flags):
    """Run as a script from a seed, the leaf hands its argv to the checkout's launcher.py: that file's own name,
    path, imports and exit status, as ``python <checkout>/launcher.py ...`` would give it, and never its own
    folder's modules (any OS; the Windows identity steps are unit-tested above)."""
    seed = tmp_path / "Program Files" / "Ouroboros" / "_internal"
    seed_leaf = seed / "ouroboros" / "launcher_windows_runtime.py"
    seed_leaf.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "ouroboros" / "launcher_windows_runtime.py", seed_leaf)
    (seed_leaf.parent / "sibling.py").write_text("NAME = 'from the seed'\n", encoding="utf-8")
    (seed_leaf.parent / "seed_only.py").write_text("", encoding="utf-8")
    checkout = tmp_path / "Ouroboros" / "repo"
    checkout.mkdir(parents=True)
    report = tmp_path / "report.json"
    (checkout / "sibling.py").write_text("NAME = 'from the checkout'\n", encoding="utf-8")
    (checkout / "launcher.py").write_text(
        "import importlib.util, json, pathlib, sys\nimport sibling\n"
        "if __name__ == '__main__':\n"
        "    pathlib.Path(sys.argv[2]).write_text(json.dumps({'argv': sys.argv, 'path0': sys.path[0],\n"
        "        'file': __file__, 'sibling': sibling.NAME, 'seed_only': bool(importlib.util.find_spec('seed_only'))}))\n"
        "    raise SystemExit(3)\n", encoding="utf-8")
    argv = [str(checkout / "launcher.py"), "--report", str(report), "--seed-bundle", str(seed)]
    result = subprocess.run([sys.executable, *flags, str(seed_leaf), *argv], capture_output=True, text=True, timeout=60)
    assert result.returncode == 3, result.stdout + result.stderr
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "argv": argv, "path0": str(checkout), "file": str(checkout / "launcher.py"), "sibling": "from the checkout",
        "seed_only": False}


def test_an_older_launchers_python_window_icon_comes_from_the_installed_app(tmp_path, monkeypatch):
    import ctypes

    asked = []
    shell = SimpleNamespace(ExtractIconW=lambda module, path, index: asked.append((module, path, index)) or 7)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(shell32=shell), raising=False)
    app = tmp_path / "Ouroboros" / "Ouroboros.exe"
    windows._offer_app_icon_to_python_windows(app)
    # pywebview 5.4 resolves windll.shell32.ExtractIconW at each window: it now finds the offered one.
    assert ctypes.windll.shell32.ExtractIconW(1, sys.executable, 0) == 7, "no installed app: Python's icon stays"
    app.parent.mkdir()
    app.touch()
    ctypes.windll.shell32.ExtractIconW(1, sys.executable, 0)
    ctypes.windll.shell32.ExtractIconW(1, str(tmp_path / "other.dll"), 2)
    assert asked == [(1, sys.executable, 0), (1, str(app), 0), (1, str(tmp_path / "other.dll"), 2)]


@pytest.mark.parametrize("mode", ["packaged", "portable_handoff", "developer", "external_ui", "not_windows"])
def test_launcher_prepares_only_packaged_windows_gui_before_opening_windows(monkeypatch, tmp_path, mode):
    import launcher
    from ouroboros import delegate_recovery

    # main() writes these globals early. Restore them without letting it acquire a real lock or start anything.
    for key in ("_headless", "_external_ui", "_external_host_update", "_external_seed_bundle", "_launch_argv", "_launch_intent"):
        monkeypatch.setattr(launcher, key, getattr(launcher, key))
    monkeypatch.setattr(launcher, "_headless", False)
    monkeypatch.setattr(launcher, "IS_WINDOWS", mode != "not_windows")
    monkeypatch.setattr(launcher, "IS_MACOS", False)
    monkeypatch.setattr(sys, "frozen", mode in {"packaged", "external_ui", "not_windows"}, raising=False)
    monkeypatch.setattr(delegate_recovery, "observe_restart_parent", lambda: {})
    argv = (["--seed-bundle", str(tmp_path)] if mode == "portable_handoff" else
            ["--no-ui"] if mode == "external_ui" else [])
    calls = []
    icon = tmp_path / "assets" / "icon.ico"

    def prepare(seed):
        calls.append(("identity", seed))
        return icon if mode == "portable_handoff" else None

    def every(window, *, decorate=None):
        calls.append(("wrapper", decorate is not None))
        if decorate:
            decorate(window)

    class EndEarly(Exception):
        pass

    def stop_before_lock():
        raise EndEarly

    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace())
    monkeypatch.setattr(launcher, "_bundle_dir", lambda: tmp_path)
    monkeypatch.setattr(launcher, "prepare_windows_desktop_identity", prepare)
    monkeypatch.setattr(launcher, "_prepare_windows_webview_runtime", lambda: (calls.append("winforms") or True, ""))
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "attach_every_window", every)
    monkeypatch.setattr(launcher, "attach_windows_window_icon", lambda _, path: calls.append(("icon", path)))
    monkeypatch.setattr(launcher, "acquire_pid_lock", stop_before_lock)
    with pytest.raises(EndEarly):
        launcher.main(argv)
    assert calls == {
        "packaged": [("identity", tmp_path), "winforms", ("wrapper", False)],
        "portable_handoff": [("identity", tmp_path), "winforms", ("wrapper", True), ("icon", icon)],
        "developer": ["winforms", ("wrapper", False)],
        "external_ui": [],
        "not_windows": [("wrapper", False)],
    }[mode]


class Hook(list):
    def __iadd__(self, callback):
        self.append(callback)
        return self


@pytest.mark.parametrize("phase", ["before_show", "already_created", "already_disposed", "load_refused"])
def test_icon_uses_real_creation_order_and_lifetime_instead_of_window_title(tmp_path, monkeypatch, caplog, phase):
    from ouroboros.launcher_context_menu import attach_every_window

    icon = tmp_path / "icon.ico"
    calls = []

    class Icon:
        def __init__(self, path):
            calls.append(("load", path))
            if phase == "load_refused":
                raise OSError("invalid icon")

        def Dispose(self):
            calls.append("dispose")

    form = SimpleNamespace(InvokeRequired=phase == "already_created", IsDisposed=phase == "already_disposed",
                           Disposed=Hook())

    def invoke(setter):
        calls.append("invoke")
        setter()

    form.Invoke = invoke
    monkeypatch.setitem(sys.modules, "System", SimpleNamespace(Action=lambda fn: fn))
    monkeypatch.setitem(sys.modules, "System.Drawing", SimpleNamespace(Icon=Icon))
    monkeypatch.setitem(sys.modules, "System.Windows.Forms", SimpleNamespace(Form=SimpleNamespace))
    monkeypatch.setattr(windows, "IS_WINDOWS", True)

    window = SimpleNamespace(native=None if phase == "before_show" else form,
                             events=SimpleNamespace(before_show=Hook()))
    webview = SimpleNamespace(create_window=lambda: window)
    attach_every_window(webview, decorate=lambda item: windows.attach_windows_window_icon(item, icon))
    assert webview.create_window() is window
    if phase == "before_show":
        assert not calls and not hasattr(form, "Icon")
        window.native = form
        window.events.before_show[-1]()
    assert calls == {
        "before_show": [("load", str(icon))],
        "already_created": ["invoke", ("load", str(icon))],
        "already_disposed": [],
        "load_refused": [("load", str(icon))],
    }[phase]
    if phase in {"already_disposed", "load_refused"}:
        assert not hasattr(form, "Icon") and not form.Disposed
        if phase == "load_refused":
            assert "kept its default icon" in caplog.text
    else:
        assert isinstance(form.Icon, Icon) and len(form.Disposed) == 1
        form.Disposed[0](form, None)
        assert calls[-1] == "dispose"
