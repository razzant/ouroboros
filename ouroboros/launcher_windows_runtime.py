"""Windows-only pythonnet/pywebview runtime preparation.

Importing ``webview`` on Windows only works after the bundled CPython DLL and
Python.Runtime.dll have been located, unblocked, and put on the DLL search path,
and after pythonnet has been pointed at the .NET Framework runtime. That whole
preparation is one concern with one caller, and it is inert everywhere else:
every entry point returns immediately off Windows. Extracted from launcher.py at
the module-size ceiling; launcher.py re-exports both names (same objects), so its
import surface and the hook names the packaging checks look for are unchanged.
"""

from __future__ import annotations

import os
import pathlib
import sys

from ouroboros.platform_layer import IS_WINDOWS


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
