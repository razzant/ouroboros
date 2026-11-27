"""Windows runtime preparation, the installed copy's identity, and stdlib-only legacy console repair.

Importing ``webview`` on Windows only works after the bundled CPython DLL and
Python.Runtime.dll have been located, unblocked, and put on the DLL search path,
and after pythonnet has been pointed at the .NET Framework runtime. The launcher
also loads this leaf by path before package adoption to check legacy handoffs:
every entry point returns immediately off Windows. Extracted from launcher.py at
the module-size ceiling; launcher.py re-exports its names (same objects), so its
import surface and the hook names the packaging checks look for are unchanged.

The per-user installer (packaging/windows/ouroboros.iss) shares three facts with
this leaf: its HKCU registration, which alone proves a copy is Setup's, the Start
shortcut's app ID, and the mutex it holds while changing the installation. Run as a
script from a Setup-owned seed, the leaf is that copy's handoff entry
(``host_source_launcher``).
"""

from __future__ import annotations

import contextlib
import fnmatch
import logging
import os
import pathlib
import sys

IS_WINDOWS = sys.platform == "win32"  # this leaf also runs by path before any package import
_INSTALLED_APP_ID = "com.ouroboros.agent"  # The per-user installer's Start-menu shortcut uses this same ID.
_SETUP_REGISTRATION = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}_is1"
_log = logging.getLogger("launcher.windows")


def _setup_registration() -> dict[str, str]:
    """Setup's HKCU uninstall values by name; empty when absent or unreadable (then no copy is Setup's)."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SETUP_REGISTRATION) as key:
            values = {}
            for name in ("Inno Setup: App Path", "InstallLocation", "UninstallString"):
                with contextlib.suppress(OSError):
                    value, _kind = winreg.QueryValueEx(key, name)
                    if isinstance(value, str):
                        values[name] = value
            return values
    except (ImportError, OSError):
        return {}


def setup_install_root(seed) -> pathlib.Path | None:
    """The per-user install root when ``seed`` is its ``_internal`` and Setup's registration owns it.

    The proof is ouroboros.iss IsOwnedInstallDir's: HKCU uninstall metadata naming that folder and an
    unins*.exe there with its .dat. The canonical path alone is not ownership: a portable ZIP unpacked
    there is foreign to Setup, has no shortcut of that app ID to relaunch it, and stays portable.
    """
    local = os.environ.get("LOCALAPPDATA") if IS_WINDOWS and seed is not None else None
    if not local:
        return None
    root = pathlib.Path(local) / "Programs" / "Ouroboros"
    try:
        if not ((root / "Ouroboros.exe").is_file() and pathlib.Path(seed).samefile(root / "_internal")):
            return None  # a portable copy elsewhere, or no per-user install at all
        registration = _setup_registration()
        recorded = (registration.get("Inno Setup: App Path") or registration.get("InstallLocation") or "").strip()
        uninstaller = pathlib.Path(registration.get("UninstallString", "").strip().strip('"'))
        owned = (bool(recorded) and pathlib.Path(recorded).samefile(root)
                 and fnmatch.fnmatch(uninstaller.name.lower(), "unins*.exe") and uninstaller.parent.samefile(root)
                 and uninstaller.is_file() and uninstaller.with_suffix(".dat").is_file())
    except OSError:
        return None  # an unreadable or vanished fact proves nothing
    return root if owned else None


def windows_source_host(seed) -> pathlib.Path | None:
    """This leaf inside a Setup-owned seed: the entry exec_source_launcher runs a checkout's launcher with."""
    if setup_install_root(seed) is None:
        return None
    host = pathlib.Path(seed) / "ouroboros" / "launcher_windows_runtime.py"
    return host if host.is_file() else None


def host_source_launcher(argv) -> None:
    """The installed copy's handoff entry: ``python <seed>/ouroboros/<this leaf> <checkout>/launcher.py ...``.

    The checkout's launcher then runs as ``python <checkout>/launcher.py ...`` would, under the Start
    shortcut's app ID and with the app's window icon, whichever release it comes from: a v7.12 launcher
    sets neither. The checkout is never rewritten.
    """
    seed = pathlib.Path(__file__).resolve().parent.parent
    script = pathlib.Path(argv[0])
    prepare_windows_desktop_identity(seed)
    _offer_app_icon_to_python_windows(seed.parent / "Ouroboros.exe")
    sys.argv[:] = [str(script), *argv[1:]]
    # The checkout's folder replaces this leaf's, whose modules would otherwise shadow top-level names.
    if sys.path and sys.path[0] and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
        sys.path[0] = str(script.parent)
    else:
        sys.path.insert(0, str(script.parent))  # an interpreter with safe_path added no script folder
    import runpy

    runpy.run_path(str(script), run_name="__main__")


def _offer_app_icon_to_python_windows(exe: pathlib.Path) -> None:
    """pywebview 5.4 gives each WinForms window ``ExtractIconW(module, sys.executable, 0)``: Python's icon here.

    A current launcher replaces that icon (``attach_windows_window_icon``); an older one does not, so this one
    request reads the installed Ouroboros.exe instead. Every other icon request passes through unchanged.
    """
    try:
        import ctypes

        shell = ctypes.windll.shell32
        extract = shell.ExtractIconW
    except (AttributeError, OSError):
        return
    python = os.path.normcase(os.path.abspath(sys.executable))

    def app_icon(module, path, index):
        if isinstance(path, str) and os.path.normcase(os.path.abspath(path)) == python and exe.is_file():
            path = str(exe)
        return extract(module, path, index)

    shell.ExtractIconW = app_icon


def prepare_windows_desktop_identity(seed: pathlib.Path) -> pathlib.Path | None:
    """Only packaged GUI callers supply the original distribution's seed, not the current checkout."""
    if not IS_WINDOWS:
        return None

    # The shortcut's target, display name and icon supply relaunch metadata. Portable copies must not
    # join that shortcut's group: the same AUMID would let Windows reopen the installed, not portable, copy.
    if setup_install_root(seed) is not None:
        try:
            _set_process_app_id(_INSTALLED_APP_ID)  # before any WinForms window or message box
        except OSError:
            _log.warning("Windows refused the Start shortcut's app identity; startup continues.", exc_info=True)

    if getattr(sys, "frozen", False):
        return None  # pywebview already extracts the icon from the packaged executable
    icon = seed / "assets" / "icon.ico"
    try:
        return icon if icon.is_file() else None
    except OSError:
        _log.warning("Windows could not locate the package's window icon.", exc_info=True)
        return None


def _set_process_app_id(app_id: str) -> None:
    import ctypes
    from ctypes import wintypes

    setter = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
    setter.argtypes = (wintypes.LPCWSTR,)
    setter.restype = ctypes.c_long  # HRESULT, signed 32-bit on Windows
    if (result := setter(app_id)) < 0:
        raise OSError(f"SetCurrentProcessExplicitAppUserModelID failed: 0x{result & 0xFFFFFFFF:08X}")


def attach_windows_window_icon(window, icon: pathlib.Path) -> None:
    """pywebview 5.4 exposes the WinForms Form in before_show; start(icon=…) only covers GTK/Qt."""
    if not IS_WINDOWS:
        return

    def apply_icon() -> None:
        try:
            from System import Action
            from System.Drawing import Icon
            from System.Windows.Forms import Form

            form = window.native
            if not isinstance(form, Form):
                return

            def set_icon() -> None:
                if form.IsDisposed:
                    return
                resource = Icon(str(icon))
                try:
                    form.Icon = resource
                    form.Disposed += lambda *_: resource.Dispose()  # retain the icon through the Form's lifetime
                except Exception:
                    resource.Dispose()
                    raise

            if form.InvokeRequired:
                form.Invoke(Action(set_icon))
            else:
                set_icon()
        except Exception:
            _log.warning("The Windows window kept its default icon.", exc_info=True)

    if getattr(window, "native", None) is not None:
        apply_icon()  # creation of a secondary pywebview window is synchronous once the GUI is running
    elif (before_show := getattr(getattr(window, "events", None), "before_show", None)) is not None:
        before_show += apply_icon


def _legacy_gui_console(seed: pathlib.Path) -> bool:
    """Evidence of an old onedir GUI launch, plus a visible console belonging only to us.

    PyInstaller leaves its archive path in the child environment. Match it to the supplied
    onedir seed and inspect the executable's PE subsystem; seed argv or a sole console client
    alone says nothing about intent. Missing facts preserve the console. Inherited/spoofed
    descriptors cannot prove exact ancestry; this is compatibility with the shipped GUI, not
    a general detector of how someone launched Python.
    """
    archive = os.environ.get("_PYI_ARCHIVE_FILE", "")
    if not archive:
        return False
    try:
        seed, image = seed.resolve(), pathlib.Path(archive).resolve()
        if seed.name != "_internal" or image.parent != seed.parent or image.suffix.lower() != ".exe":
            return False
        with image.open("rb") as binary:
            if binary.read(2) != b"MZ":
                return False
            binary.seek(60)
            offset = binary.read(4)
            if len(offset) != 4:
                return False
            binary.seek(int.from_bytes(offset, "little"))
            header = binary.read(94)  # PE signature + COFF header + optional header through Subsystem
        if header[:4] != b"PE\0\0" or header[24:26] not in (b"\x0b\x01", b"\x0b\x02") or header[92:94] != b"\x02\0":
            return False  # IMAGE_SUBSYSTEM_WINDOWS_GUI, both PE32 and PE32+
        import ctypes
        from ctypes import wintypes

        kernel, user = ctypes.WinDLL("kernel32"), ctypes.WinDLL("user32")
        kernel.GetConsoleProcessList.argtypes = (ctypes.POINTER(wintypes.DWORD), wintypes.DWORD)
        kernel.GetConsoleProcessList.restype = wintypes.DWORD
        kernel.GetConsoleWindow.argtypes = ()
        kernel.GetConsoleWindow.restype = wintypes.HWND
        user.IsWindowVisible.argtypes = (wintypes.HWND,)
        user.IsWindowVisible.restype = wintypes.BOOL
        clients = (wintypes.DWORD * 2)()
        window = kernel.GetConsoleWindow()
        return (kernel.GetConsoleProcessList(clients, 2) == 1 and clients[0] == os.getpid()
                and bool(window and user.IsWindowVisible(window)))
    except (OSError, ValueError):
        return False


def rehome_legacy_handoff(args) -> None:
    """Called by launcher.py's guarded, stdlib-only prologue; never FreeConsole-and-continue."""
    if not IS_WINDOWS:
        return
    seed = next((arg.partition("=")[2] if "=" in arg else args[index + 1] if index + 1 < len(args) else ""
                 for index, arg in enumerate(args) if arg.split("=", 1)[0] == "--seed-bundle"), "")
    if not seed or not _legacy_gui_console(pathlib.Path(seed)):
        return
    import subprocess

    try:
        subprocess.Popen([sys.executable, *sys.orig_argv[1:]], close_fds=True,
                         creationflags=subprocess.CREATE_NO_WINDOW, env={**os.environ, "OUROBOROS_LAUNCHER_HANDOFF": "1"})
        os._exit(0)
    except OSError as exc:
        print(f"Ouroboros could not leave this console window: {exc}", file=sys.stderr, flush=True)


_windows_dll_dir_handles: list = []


# ouroboros.iss SetupChangingName: held exclusively by Setup or Uninstall while changing the install.
_SETUP_CHANGING = "Local\\Ouroboros {6FB7704B-C2C5-4252-A5D0-BEC473FD007D} changing"


def _setup_root() -> str:
    """Setup's {app}: Programs\\Ouroboros in the known local app-data folder (FOLDERID_LocalAppData), not the env var."""
    import ctypes
    import uuid

    path = ctypes.c_wchar_p()
    folder = ctypes.create_string_buffer(uuid.UUID("F1B32785-6FBA-4FCF-9D55-7B8E7F157091").bytes_le, 16)
    if ctypes.windll.shell32.SHGetKnownFolderPath(folder, 0, None, ctypes.byref(path)) != 0:
        return ""
    try:
        return os.path.join(path.value, "Programs", "Ouroboros")
    finally:
        ctypes.windll.ole32.CoTaskMemFree(path)


def refuse_while_setup_changes(executable, args=(), *, console=False) -> None:
    """Exit 75 before package work when Setup or Uninstall holds ``_SETUP_CHANGING``.

    Only programs inside the per-user install ask (portable copies are unaffected). GUI starts exit
    immediately with no dialog so Start clicks cannot keep ``Ouroboros.exe`` mapped; CLI prints to stderr.
    """
    del args  # kept in signature for launcher.py
    if not IS_WINDOWS:
        return
    root = _setup_root()
    if not root or not os.path.normcase(os.path.abspath(executable)).startswith(os.path.normcase(root) + os.sep):
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenMutexW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.OpenMutexW.restype = wintypes.HANDLE
    held = kernel.OpenMutexW(0x00100000, False, _SETUP_CHANGING)  # SYNCHRONIZE: existence check only
    if held:
        kernel.CloseHandle(held)
    elif ctypes.get_last_error() == 2:  # ERROR_FILE_NOT_FOUND: no active change; other errors fail closed
        return
    if console:
        print(
            "ouroboros: Ouroboros Setup or Uninstall is changing this installation. "
            "Try again once it has finished.",
            file=sys.stderr,
            flush=True,
        )
    raise SystemExit(75)


def _show_windows_message(title: str, message: str) -> None:
    if not IS_WINDOWS:
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:
        pass


def _prepare_windows_webview_runtime() -> tuple[bool, str]:
    """Prepare pythonnet/pywebview runtime before importing webview on Windows."""
    if not IS_WINDOWS:
        return True, ""

    base_dir = pathlib.Path(getattr(sys, "_MEIPASS", pathlib.Path(sys.executable).parent))
    exe_dir = pathlib.Path(sys.executable).parent
    runtime_dir = base_dir / "pythonnet" / "runtime"
    webview_lib_dir = base_dir / "webview" / "lib"
    py_dll_name = f"python{sys.version_info[0]}{sys.version_info[1]}.dll"

    def _unblock_file(path: pathlib.Path) -> None:
        try:
            os.remove(f"{path}:Zone.Identifier")
        except OSError:
            pass

    def _unblock_tree(root: pathlib.Path) -> None:
        if not root.is_dir():
            return
        for child in root.rglob("*"):
            if child.is_file() and child.suffix.lower() in {".dll", ".exe", ".pyd"}:
                _unblock_file(child)

    py_dll_candidates = [
        base_dir / py_dll_name,
        exe_dir / py_dll_name,
    ]
    for root, _dirs, files in os.walk(base_dir):
        if py_dll_name in files:
            py_dll_candidates.append(pathlib.Path(root) / py_dll_name)
            if len(py_dll_candidates) >= 6:
                break

    py_dll_path = next((path for path in py_dll_candidates if path.is_file()), None)
    runtime_dll_path = runtime_dir / "Python.Runtime.dll"
    if not runtime_dll_path.is_file():
        for root, _dirs, files in os.walk(base_dir):
            if "Python.Runtime.dll" in files:
                runtime_dll_path = pathlib.Path(root) / "Python.Runtime.dll"
                break

    if py_dll_path is None:
        return False, f"Bundled {py_dll_name} was not found."
    if not runtime_dll_path.is_file():
        return False, "Bundled Python.Runtime.dll was not found."

    _unblock_file(py_dll_path)
    _unblock_file(runtime_dll_path)
    _unblock_tree(runtime_dll_path.parent)
    _unblock_tree(webview_lib_dir)

    os.environ["PYTHONNET_RUNTIME"] = "netfx"
    os.environ["PYTHONNET_PYDLL"] = str(py_dll_path)

    search_dirs = []
    for candidate in (
        base_dir,
        exe_dir,
        runtime_dir,
        runtime_dll_path.parent,
        py_dll_path.parent,
        webview_lib_dir,
    ):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in search_dirs:
            search_dirs.append(candidate_str)

    current_path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    os.environ["PATH"] = os.pathsep.join(search_dirs + [part for part in current_path_parts if part and part not in search_dirs])

    if hasattr(os, "add_dll_directory"):
        global _windows_dll_dir_handles
        for candidate in search_dirs:
            try:
                _windows_dll_dir_handles.append(os.add_dll_directory(candidate))
            except (FileNotFoundError, OSError):
                pass

    try:
        from clr_loader import get_netfx
        from pythonnet import set_runtime

        set_runtime(get_netfx())
    except Exception as exc:
        return False, f"Windows .NET runtime init failed: {exc}"

    return True, ""


if __name__ == "__main__":  # exec_source_launcher's entry for a Setup-owned copy (windows_source_host)
    host_source_launcher(sys.argv[1:])
