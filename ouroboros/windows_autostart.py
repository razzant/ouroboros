"""Windows logon autostart for the packaged desktop installation.

Owner decision: autostart is the HKCU ``Run`` registry key (NOT a shortcut in
``shell:startup``). The launcher alone knows the frozen ``launcher.exe`` path,
so it exports ``OUROBOROS_LAUNCHER_EXE`` to the managed server; this module is
the single reader and the single writer of the key. The key's value is the
quoted executable path of THIS installation, and truth is the registry itself:
the GET state reads the key, not a settings.json mirror — a key edited by hand
or by uninstall in Windows' own tooling is reported as it is, and enable is
idempotent (re-pointing the value at the current installation).
Non-Windows or unfrozen runs have no autostart: the endpoint reports
``available: false`` and never touches the registry. No background service is
created; Windows Explorer runs the key at logon exactly as it would a manual
double-click. Registry failures are typed errors surfaced to the owner, never
swallowed into a pretend-enabled state.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE_NAME = "Ouroboros"

# winreg is Windows-only; importing it lazily keeps this module importable
# (and its functions safely unavailable) on every platform.
_winreg: Any = None
try:  # pragma: no cover - trivial import guard
    import winreg as _winreg_module  # type: ignore

    _winreg = _winreg_module
except ImportError:  # pragma: no cover - non-Windows
    _winreg = None


def _launcher_exe() -> Optional[str]:
    """The frozen launcher path this installation exported, or None."""
    raw = str(os.environ.get("OUROBOROS_LAUNCHER_EXE") or "").strip()
    return raw or None


def autostart_available() -> bool:
    """True only on a packaged Windows desktop run: winreg present + launcher exe known."""
    return _winreg is not None and sys_platform_is_windows() and _launcher_exe() is not None


def sys_platform_is_windows() -> bool:
    import sys

    return sys.platform == "win32"


def read_autostart_state() -> Dict[str, Any]:
    """The registry-truth state for the owner surface (never raises)."""
    available = autostart_available()
    state: Dict[str, Any] = {
        "available": available,
        "platform_supported": _winreg is not None and sys_platform_is_windows(),
    }
    if not available:
        state.update({"enabled": False, "launcher_exe": None, "reason": _unavailable_reason()})
        return state
    try:
        with _winreg.OpenKey(
            _winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, _winreg.KEY_READ
        ) as key:
            value, _value_type = _winreg.QueryValueEx(key, RUN_VALUE_NAME)
            # The stored form is the quoted invocation; surface the bare path.
            raw = str(value).strip()
            if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            state["enabled"] = True
            state["launcher_exe"] = raw
            state["value_name"] = RUN_VALUE_NAME
            state["key_path"] = RUN_KEY_PATH
            return state
    except FileNotFoundError:
        state.update({"enabled": False, "launcher_exe": _launcher_exe(),
                      "value_name": RUN_VALUE_NAME, "key_path": RUN_KEY_PATH})
        return state
    except OSError as exc:
        # Registry unreadable: report disabled with the reason, keeping the
        # response contract's boolean shape; never a silent "enabled".
        state.update({"enabled": False, "launcher_exe": None,
                      "reason": f"registry read failed: {exc}"})
        return state


def set_autostart(enabled: bool) -> Dict[str, Any]:
    """Create or delete the HKCU Run key. Returns the post-write registry truth."""
    if not autostart_available():
        return {"ok": False, "error": _unavailable_reason()}
    exe = _launcher_exe()
    try:
        if enabled:
            # SetStringValue: REG_SZ, matching how Windows tooling renders Run
            # entries. The quoted form matches the conventional double-click
            # invocation and survives paths with spaces.
            with _winreg.OpenKey(
                _winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, _winreg.KEY_SET_VALUE
            ) as key:
                _winreg.SetValueEx(key, RUN_VALUE_NAME, 0, _winreg.REG_SZ, f'"{exe}"')
        else:
            try:
                with _winreg.OpenKey(
                    _winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, _winreg.KEY_SET_VALUE
                ) as key:
                    _winreg.DeleteValue(key, RUN_VALUE_NAME)
            except FileNotFoundError:
                # Disabling an absent key is idempotent success.
                pass
    except OSError as exc:
        return {"ok": False, "error": f"registry write failed: {exc}"}
    state = read_autostart_state()
    if state.get("reason") and not state.get("enabled") and enabled:
        # The write reported success but the post-write read failed: do not
        # stamp ok over an unverifiable state.
        return {"ok": False, "error": f"registry written but re-read failed: {state['reason']}"}
    state["ok"] = True
    return state


def _unavailable_reason() -> str:
    if _winreg is None or not sys_platform_is_windows():
        return "autostart is only available on Windows desktop installations"
    if _launcher_exe() is None:
        return (
            "OUROBOROS_LAUNCHER_EXE is not set; autostart requires the packaged "
            "desktop launcher"
        )
    return "autostart unavailable"
