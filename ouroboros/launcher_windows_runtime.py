"""Windows runtime preparation and stdlib-only legacy console repair.

Importing ``webview`` on Windows only works after the bundled CPython DLL and
Python.Runtime.dll have been located, unblocked, and put on the DLL search path,
and after pythonnet has been pointed at the .NET Framework runtime. The launcher
also loads this leaf by path before package adoption to check legacy handoffs:
every entry point returns immediately off Windows. Extracted from launcher.py at
the module-size ceiling; launcher.py re-exports both names (same objects), so its
import surface and the hook names the packaging checks look for are unchanged.
"""

from __future__ import annotations

import os
import pathlib
import sys

IS_WINDOWS = sys.platform == "win32"  # this leaf also runs by path before any package import


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
