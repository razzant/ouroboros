"""Start the packaged Windows desktop app when the owner signs in.

The per-user ``Run`` value named ``Ouroboros`` is the only state — no settings.json
mirror, because Windows' own Startup apps page edits the same entry between reads.
That page keeps its switch in ``StartupApproved\\Run`` (odd first byte = switched
off); turning autostart on or off clears the switch, so Windows follows the new
entry. The value name is the app's own: turning on retargets an entry left by
another copy of Ouroboros, turning off removes it.

The entry passes the launcher's existing ``--launch-intent automatic``, so a
sign-in never starts a copy the owner stopped with Panic
(``launcher_bootstrap.automatic_launch_allowed``); an owner start still resumes.
The target is ``Ouroboros.exe`` beside the PyInstaller ``_internal`` bundle that
the launcher exports to its managed server (``OUROBOROS_BUNDLE_DIR``); source,
headless and non-Windows runs report ``unavailable`` and never touch the registry.
"""

from __future__ import annotations

import ntpath
import os
import pathlib
from typing import Any, Literal, Optional

from ouroboros.platform_layer import BUNDLE_DIR_ENV, IS_WINDOWS

AutostartState = Literal["unavailable", "off", "on", "other_copy", "disabled_by_os"]

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"
VALUE_NAME = "Ouroboros"
LAUNCHER_EXE = "Ouroboros.exe"  # the EXE name in Ouroboros.spec
SIGN_IN_ARGS = "--launch-intent automatic"


def _winreg() -> Any:
    """The guarded function-local platform import (platform abstraction rule)."""
    if not IS_WINDOWS:
        raise OSError("the Windows registry exists only on Windows")
    import winreg  # type: ignore[import-not-found]

    return winreg


def launcher_path() -> Optional[pathlib.Path]:
    """The packaged desktop launcher that started this server, or None."""
    if not IS_WINDOWS or os.environ.get("OUROBOROS_PRESENTATION") != "desktop_window":
        return None
    bundle = str(os.environ.get(BUNDLE_DIR_ENV) or "").strip()
    if not bundle or pathlib.Path(bundle).name.lower() != "_internal":
        return None
    exe = pathlib.Path(bundle).parent / LAUNCHER_EXE
    return exe if exe.is_file() else None


def sign_in_command(exe: pathlib.Path) -> str:
    """The exact Run command this copy writes and recognises as its own."""
    return f'"{exe}" {SIGN_IN_ARGS}'


def _is_own_command(command: object, exe: pathlib.Path) -> bool:
    """Windows paths compare without case; the launcher's arguments do not."""
    if not isinstance(command, str):
        return False
    target, _, args = command.strip().partition('" ')
    return ntpath.normcase(target + '"') == ntpath.normcase(f'"{exe}"') and args == SIGN_IN_ARGS


def _read_value(key_path: str) -> object:
    winreg = _winreg()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            return winreg.QueryValueEx(key, VALUE_NAME)[0]
    except FileNotFoundError:
        return None


def _delete_value(key_path: str) -> None:
    winreg = _winreg()
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        pass


def autostart_state() -> AutostartState:
    """What Windows will do at the next sign-in for THIS copy of the app."""
    exe = launcher_path()
    if exe is None:
        return "unavailable"
    command = _read_value(RUN_KEY)
    if command is None:
        return "off"
    if not _is_own_command(command, exe):
        return "other_copy"
    switch = _read_value(APPROVED_KEY)
    if isinstance(switch, bytes) and switch[:1] and switch[0] & 1:
        return "disabled_by_os"
    return "on"


def follow_moved_app(exe: pathlib.Path) -> bool:
    """Point this app's own Run entry (its value name, its sign-in arguments) at ``exe`` when the program it
    names is gone, keeping the StartupApproved switch (``set_autostart`` would clear it); a live other copy
    stays ``other_copy``."""
    command = _read_value(RUN_KEY)
    target, _sep, args = command.strip().partition('" ') if isinstance(command, str) else ("", "", "")
    program = target[1:] if target.startswith('"') else ""
    if args != SIGN_IN_ARGS or not program or ntpath.normcase(program) == ntpath.normcase(str(exe)) or os.path.exists(program):
        return False
    winreg = _winreg()
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, sign_in_command(exe))
    return True


def set_autostart(enabled: bool) -> AutostartState:
    """Write or remove the entry; raises OSError when the registry refuses."""
    exe = launcher_path()
    if exe is None:
        return "unavailable"
    if enabled:
        winreg = _winreg()
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, sign_in_command(exe))
    else:
        _delete_value(RUN_KEY)
    _delete_value(APPROVED_KEY)
    return autostart_state()
