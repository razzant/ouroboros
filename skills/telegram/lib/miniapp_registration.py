"""Registration surface for the automatic Telegram Mini App gateway."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from ..scripts.platform_support import machine_architecture, process_alive
from ..scripts.telegram_settings import owner_chat_id


_CONFIG_SCHEMA = 2
_CONFIG_NAME = "runtime_config.json"
_STATUS_NAME = "status.json"


class ConfigurationError(RuntimeError):
    pass


def _bounded_text(value: Any, maximum: int = 200) -> str:
    return str(value or "").strip().replace("\r", " ").replace("\n", " ")[:maximum]


def _platform_error() -> str:
    system = platform.system()
    machine = machine_architecture()
    arch = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "amd64",
        "amd64": "amd64",
    }.get(machine)
    supported = (
        (system == "Darwin" and arch in {"arm64", "amd64"})
        or (system == "Linux" and arch in {"arm64", "amd64"})
        or (system == "Windows" and arch == "amd64")
    )
    if supported:
        return ""
    return (
        "Telegram Mini App does not have a pinned cloudflared asset for "
        f"{system or 'unknown'}/{machine or 'unknown'}."
    )


def _state_path(api: Any, name: str) -> Path:
    raw_root = Path(api.get_state_dir()).expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_root.is_symlink():
        raise ConfigurationError("Skill state directory is an unsafe link.")
    root = raw_root.resolve(strict=True)
    path = root / name
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError("Skill state path escaped its private directory.") from exc
    if path.is_symlink():
        raise ConfigurationError("Skill state file is an unsafe link.")
    return path


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(encoded.encode("utf-8")) > 16_384:
        raise ConfigurationError("Runtime configuration is unexpectedly large.")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(encoded, encoding="utf-8")
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigurationError("Could not persist Mini App runtime configuration.") from exc


def _seed_starting_status(api: Any) -> None:
    """Give registration-without-spawn a heartbeat-bounded visible state."""

    path = _state_path(api, _STATUS_NAME)
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError, TypeError):
        current = None
    if isinstance(current, dict):
        try:
            age = int(time.time()) - int(current.get("updated_at_epoch") or 0)
            pid = int(current.get("pid") or 0)
        except (TypeError, ValueError):
            age = 999
            pid = 0
        if -60 <= age <= 45 and pid > 1 and process_alive(pid):
            return
    _atomic_json(
        path,
        {
            "schema": 2,
            "state": "starting",
            "reason_code": "registered",
            "message": "Waiting for the Mini App companion to publish its first heartbeat.",
            "pid": 0,
            "updated_at_epoch": int(time.time()),
        },
    )


def _write_runtime_config(api: Any) -> dict[str, Any]:
    info = api.get_runtime_info()
    try:
        core_port = int(info.get("server_port") or 0)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Ouroboros server port is invalid.") from exc
    if not 1 <= core_port <= 65_535:
        raise ConfigurationError("Ouroboros server port is unavailable.")
    payload = {
        "schema": _CONFIG_SCHEMA,
        "core_port": core_port,
        "owner_chat_id": owner_chat_id(Path(api.get_state_dir())),
        "button_text": "Ouroboros",
        "tunnel": "cloudflare_quick",
    }
    _atomic_json(_state_path(api, _CONFIG_NAME), payload)
    return payload


def _read_status(api: Any) -> dict[str, Any]:
    try:
        path = _state_path(api, _STATUS_NAME)
    except ConfigurationError:
        return {"state": "error", "message": "Companion status path is unsafe."}
    if not path.is_file() or path.is_symlink():
        return {
            "state": "starting",
            "message": "The companion has not published status yet.",
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"state": "error", "message": "Companion status is unreadable."}
    if not isinstance(value, dict):
        return {"state": "error", "message": "Companion status is invalid."}
    # Return only the documented, bounded public diagnostics.  In particular,
    # never echo environment, headers, initData, cookies, or Telegram API text.
    result = {
        "state": _bounded_text(value.get("state"), 32) or "unknown",
        "message": _bounded_text(value.get("message"), 300),
    }
    try:
        pid = int(value.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    stale_status = False
    terminal = {"stopped", "rollback_pending", "error", "unavailable", "disabled"}
    if result["state"] not in terminal:
        registration_pending = (
            pid == 0
            and result["state"] == "starting"
            and _bounded_text(value.get("reason_code"), 64) == "registered"
        )
        # The companion is never PID 1: it is always the server's child.
        alive = registration_pending or (pid > 1 and process_alive(pid))
        try:
            updated_at = int(value.get("updated_at_epoch") or 0)
        except (TypeError, ValueError):
            updated_at = 0
        age = int(time.time()) - updated_at
        heartbeat_fresh = updated_at > 0 and -60 <= age <= 45
        if not alive or not heartbeat_fresh:
            stale_status = True
            result = {
                "state": "stale",
                "message": (
                    "The companion heartbeat is stale or its process is no longer running."
                ),
                "reason_code": "heartbeat_stale",
            }
    public_url = _bounded_text(value.get("public_url"), 2048)
    if (
        result["state"] == "ready"
        and public_url.startswith("https://")
        and public_url.endswith(".trycloudflare.com/")
    ):
        result["public_url"] = public_url
    version = _bounded_text(value.get("cloudflared_version"), 64)
    if version:
        result["cloudflared_version"] = version
    for key, maximum in (
        ("instance_id", 64),
        ("platform", 96),
    ):
        bounded = _bounded_text(value.get(key), maximum)
        if bounded:
            result[key] = bounded
    if not stale_status:
        reason_code = _bounded_text(value.get("reason_code"), 64)
        if reason_code:
            result["reason_code"] = reason_code
    for key in ("updated_at_epoch", "last_ready_at_epoch", "attempt", "next_retry_at_epoch"):
        try:
            result[key] = max(0, int(value.get(key) or 0))
        except (TypeError, ValueError):
            pass
    security = value.get("security")
    if isinstance(security, dict):
        allowed = {
            "auth_success",
            "auth_rejected",
            "auth_rate_limited",
            "auth_busy",
            "active_sessions",
            "active_websockets",
        }
        safe_security: dict[str, int] = {}
        for key in sorted(allowed):
            try:
                safe_security[key] = max(0, min(int(security.get(key) or 0), 2_147_483_647))
            except (TypeError, ValueError):
                continue
        if safe_security:
            result["security"] = safe_security
    return result


def register(api: Any) -> bool:
    """Register the Mini App companion; False when this platform has none (the text bridge stays)."""
    _write_runtime_config(api)
    platform_error = _platform_error()
    if platform_error:
        _atomic_json(
            _state_path(api, _STATUS_NAME),
            {
                "schema": 2,
                "state": "unavailable",
                "reason_code": "unsupported_platform",
                "message": platform_error,
                "platform": f"{platform.system() or 'unknown'}/{machine_architecture() or 'unknown'}",
                "pid": 0,
                "updated_at_epoch": int(time.time()),
            },
        )
        api.log("warning", platform_error + " Text bridge remains active.")
        return False
    _seed_starting_status(api)
    api.register_companion_process("miniapp_gateway")
    api.log(
        "info",
        "telegram registered automatic owner-only Mini App gateway",
    )
    return True


__all__ = ["ConfigurationError", "_read_status", "register"]
