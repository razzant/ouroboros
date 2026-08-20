"""Connection registry and backend selection for the unix_computer_use skill.

Verbatim extraction from ``plugin.py`` (v7 stream W). ``_ComputerUse`` mixes
this class in, so every method keeps its exact name, signature and body; the
registry is the single owner of ``connections.json`` and the active-connection
pointer, including the fail-closed rule that a non-local active connection is
never silently served by the local desktop.
"""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from typing import Any

from .cu_runtime import (
    _ACTIVE_CONNECTION_FILE,
    _CONNECTIONS_FILE,
    _REMOTE_BACKENDS,
    _json,
)


class _ConnectionRegistryMixin:
    """Connection registry, active-connection resolution and the connection tools."""

    def _connections_path(self) -> pathlib.Path:
        return self.state_dir / _CONNECTIONS_FILE

    def _active_connection_path(self) -> pathlib.Path:
        return self.state_dir / _ACTIVE_CONNECTION_FILE

    def _read_connections(self) -> dict[str, Any]:
        """Read connection registry; always includes local default."""
        data: dict[str, Any] = {"active": "local", "connections": {"local": {"backend": "local", "enabled": True}}}
        try:
            raw = json.loads(self._connections_path().read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                conns = raw.get("connections")
                if isinstance(conns, dict):
                    data["connections"].update({str(k): v for k, v in conns.items() if isinstance(v, dict)})
                active = str(raw.get("active") or "").strip()
                if active:
                    data["active"] = active
        except Exception:
            pass
        try:
            active_file = self._active_connection_path().read_text(encoding="utf-8").strip()
            if active_file:
                data["active"] = active_file
        except Exception:
            pass
        data["connections"].setdefault("local", {"backend": "local", "enabled": True})
        # Unknown active name is PRESERVED (not reset to local); _active_connection fails it closed.
        return data

    def _atomic_write(self, path: pathlib.Path, text: str) -> None:
        """Write+rename: a crash can't leave a torn registry file (which could route remote→local)."""
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def _write_connections(self, data: dict[str, Any]) -> None:
        data.setdefault("connections", {})
        data["connections"].setdefault("local", {"backend": "local", "enabled": True})
        # Registry first, active pointer last: a lost second write still names a live connection.
        self._atomic_write(self._connections_path(), json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        try:
            self._atomic_write(self._active_connection_path(), str(data.get("active") or "local"))
        except Exception:
            pass

    def _active_connection(self) -> tuple[str, dict[str, Any]]:
        data = self._read_connections()
        name = str(data.get("active") or "local")
        conn = dict((data.get("connections") or {}).get(name) or {})
        if name == "local":
            return name, (conn or {"backend": "local", "enabled": True})
        # FAIL CLOSED: any NON-local active connection that is missing from the
        # registry (corrupt connections.json), disabled, or carries an unknown
        # backend is marked disabled — it must NEVER fall back to the local
        # desktop. _is_remote() below still returns True for such a name, so the
        # input tools route into _remote_pyautogui (which refuses on "disabled")
        # rather than silently driving the host.
        backend = str(conn.get("backend") or "").strip().lower()
        if not conn or backend not in _REMOTE_BACKENDS or not conn.get("enabled", True):
            marker = {**conn, "backend": backend or "unknown", "disabled": True}
            if not conn:
                marker["missing"] = True
            return name, marker
        return name, conn

    def _disabled_connection_error(self, name: str, conn: dict[str, Any]) -> str:
        return _json({
            "ok": False, "connection": name, "backend": str(conn.get("backend") or "local"),
            "error": f"active connection {name!r} is unusable (disabled or unknown backend); re-add it via add_connection or switch with use_local/activate_connection",
        })

    def _active_backend_name(self) -> str:
        _name, conn = self._active_connection()
        return str(conn.get("backend") or "local").strip().lower() or "local"

    def _is_remote(self) -> bool:
        # Any non-local ACTIVE name is "remote" for dispatch purposes: usable
        # remotes act on the VM; unusable ones (disabled/missing/unknown) are
        # refused in the remote path — never silently handled locally.
        name, _conn = self._active_connection()
        return name != "local"

    def list_connections(self) -> str:
        data = self._read_connections()
        active = str(data.get("active") or "local")
        safe: dict[str, Any] = {"active": active, "connections": {}}
        for name, conn in (data.get("connections") or {}).items():
            if not isinstance(conn, dict):
                continue
            c = {k: v for k, v in conn.items() if "key" not in str(k).lower() and "secret" not in str(k).lower()}
            c["active"] = name == active
            safe["connections"][name] = c
        return _json({"ok": True, **safe})

    def add_connection(self, *, name: str, backend: str, target: str = "", target_file: str = "",
                       host: str = "", user: str = "", port: int = 22,
                       ssh_alias: str = "", enabled: bool = True, activate: bool = False) -> str:
        """Add/update a connection. Does not accept or store private keys."""
        name = str(name or "").strip()
        backend = str(backend or "").strip().lower()
        if not name or name == "local":
            return _json({"ok": False, "error": "name is required and cannot be 'local'"})
        if backend not in {"osworld_http", "ssh_macos"}:
            return _json({"ok": False, "error": "backend must be one of: osworld_http, ssh_macos"})
        conn: dict[str, Any] = {"backend": backend, "enabled": bool(enabled)}
        if backend == "osworld_http":
            if target:
                conn["target"] = str(target).strip().rstrip("/")
            if target_file:
                conn["target_file"] = str(target_file).strip()
            if not conn.get("target") and not conn.get("target_file"):
                return _json({"ok": False, "error": "osworld_http requires target or target_file"})
        if backend == "ssh_macos":
            if ssh_alias:
                conn["ssh_alias"] = str(ssh_alias).strip()
            else:
                if not host:
                    return _json({"ok": False, "error": "ssh_macos requires host or ssh_alias"})
                conn.update({"host": str(host).strip(), "user": str(user or "").strip(), "port": int(port or 22)})
        data = self._read_connections()
        data.setdefault("connections", {})[name] = conn
        if activate:
            data["active"] = name
        self._write_connections(data)
        return _json({"ok": True, "connection": name, "backend": backend, "active": data.get("active") == name})

    def activate_connection(self, *, name: str) -> str:
        name = str(name or "").strip()
        data = self._read_connections()
        if name not in data.get("connections", {}):
            return _json({"ok": False, "error": f"unknown connection {name!r}"})
        data["active"] = name
        self._write_connections(data)
        return _json({"ok": True, "active": name, "connection": data["connections"][name]})

    def use_local(self) -> str:
        data = self._read_connections()
        data["active"] = "local"
        self._write_connections(data)
        return _json({"ok": True, "active": "local"})

    def clear_active_connection(self) -> str:
        return self.use_local()

    def test_connection(self, *, name: str = "") -> str:
        if name:
            data = self._read_connections()
            conn = dict((data.get("connections") or {}).get(str(name)) or {})
            if not conn:
                return _json({"ok": False, "error": f"unknown connection {name!r}"})
            conn_name = str(name)
        else:
            conn_name, conn = self._active_connection()
        backend = str(conn.get("backend") or "local").lower()
        if backend == "local":
            return _json({"ok": True, "connection": conn_name, "backend": "local", **self._capabilities()})
        if backend == "osworld_http":
            return self._test_osworld(conn, conn_name)
        if backend == "ssh_macos":
            return self._test_ssh_macos(conn, conn_name)
        return _json({"ok": False, "connection": conn_name, "error": f"unsupported backend {backend!r}"})
