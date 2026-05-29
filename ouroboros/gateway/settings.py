"""Settings, onboarding, and Claude-runtime gateway endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import socket
import sys
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from ouroboros.config import (
    DATA_DIR,
    SETTINGS_DEFAULTS as _SETTINGS_DEFAULTS,
    apply_settings_to_env as _apply_settings_to_env,
    load_settings,
    save_settings,
)
from ouroboros.gateway._helpers import json_error, json_exception, request_drive_root
from ouroboros.onboarding_wizard import build_onboarding_html
from ouroboros.platform_layer import is_container_env
from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    classify_runtime_provider_change,
    has_startup_ready_provider,
    has_supervisor_provider,
)
from ouroboros.settings_setup_contract import (
    BUDGET_SETTING_KEYS,
    build_setup_contract,
    parse_budget_setting,
)
from ouroboros.utils import append_jsonl, atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)
DEFAULT_PORT = int(os.environ.get("OUROBOROS_SERVER_PORT", "8765"))

_SECRET_SETTING_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "OUROBOROS_NETWORK_PASSWORD",
}
_CUSTOM_SECRET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")

def _get_lan_ip() -> str:
    """Return LAN IP via UDP socket trick; no packet is sent."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 80))  # RFC 5737 TEST-NET-1, no packet sent
            return s.getsockname()[0]
    except OSError:
        return ""


_WILDCARD_HOSTS = frozenset({"0.0.0.0", ""})


def _is_wildcard_host(host: str) -> bool:
    return host in _WILDCARD_HOSTS


def _trust_nonlocal_bind_without_password_enabled() -> bool:
    raw = os.environ.get("OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD", "")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_network_meta(bind_host: str, bind_port: int) -> dict:
    """Build /api/settings network metadata."""
    from ouroboros.server_auth import get_network_auth_startup_warning, is_loopback_host
    # Strip IPv6 brackets before loopback classification.
    unbracketed = bind_host[1:-1] if bind_host.startswith("[") and bind_host.endswith("]") else bind_host
    loopback = is_loopback_host(unbracketed)
    if loopback:
        return {
            "bind_host": bind_host,
            "bind_port": bind_port,
            "lan_ip": "",
            "reachability": "loopback_only",
            "recommended_url": "",
            "warning": "Server is bound to localhost — not accessible from other devices.",
        }
    wildcard = _is_wildcard_host(bind_host)
    if wildcard:
        if is_container_env():
            lan_ip = ""
        else:
            lan_ip = _get_lan_ip()
    elif bind_host in ("::", "[::]"):
        # AF_INET startup cannot advertise an IPv6 wildcard LAN IP reliably.
        lan_ip = ""
    else:
        # Use unbracketed form so URL construction can re-bracket IPv6 uniformly.
        lan_ip = unbracketed

    auth_warning = get_network_auth_startup_warning(bind_host) or ""
    if lan_ip:
        host_in_url = f"[{lan_ip}]" if ":" in lan_ip else lan_ip
        reachability = "lan_reachable"
        recommended_url = f"http://{host_in_url}:{bind_port}"
        warning = auth_warning
    else:
        reachability = "host_ip_unknown"
        recommended_url = f"http://your-host-ip:{bind_port}"
        warning = " ".join(
            part for part in [
                "Could not detect LAN IP automatically." if wildcard else "",
                auth_warning,
            ]
            if part
        )
    return {
        "bind_host": bind_host,
        "bind_port": bind_port,
        "lan_ip": lan_ip,
        "reachability": reachability,
        "recommended_url": recommended_url,
        "warning": warning,
    }


def _mask_secret_value(value: Any) -> str:
    text = str(value or "")
    return text[:8] + "..." if len(text) > 8 else "***"


def _looks_masked_secret(value: Any) -> bool:
    text = str(value or "").strip()
    return text == "***" or text.endswith("...")


def _mask_mcp_servers_payload(servers: Any) -> list:
    if not isinstance(servers, list):
        return []
    try:
        from ouroboros.mcp_client import canonical_server_id as _mcp_canonical_id
    except Exception:
        _mcp_canonical_id = lambda value: str(value or "").strip()  # type: ignore[assignment]
    out = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        clone = dict(entry)
        if clone.get("id"):
            clone["id"] = _mcp_canonical_id(clone.get("id"))
        token = str(clone.get("auth_token") or "")
        if token:
            clone["auth_token"] = _mask_secret_value(token)
            clone["auth_configured"] = True
        else:
            clone["auth_token"] = ""
            clone["auth_configured"] = False
        out.append(clone)
    return out


def _rehydrate_mcp_servers_payload(incoming: Any, current: Any) -> list:
    if not isinstance(incoming, list):
        return []
    try:
        from ouroboros.mcp_client import canonical_server_id as _mcp_canonical_id
    except Exception:
        _mcp_canonical_id = lambda value: str(value or "").strip()  # type: ignore[assignment]
    current_by_id: Dict[str, Dict[str, Any]] = {}
    if isinstance(current, list):
        for entry in current:
            if isinstance(entry, dict):
                cur_id = _mcp_canonical_id(entry.get("id"))
                if cur_id:
                    current_by_id[cur_id] = entry
    out = []
    for entry in incoming:
        if not isinstance(entry, dict):
            continue
        clone = dict(entry)
        clone.pop("auth_configured", None)
        if clone.get("id"):
            clone["id"] = _mcp_canonical_id(clone.get("id"))
        token = str(clone.get("auth_token") or "")
        if _looks_masked_secret(token):
            existing = current_by_id.get(_mcp_canonical_id(clone.get("id")))
            clone["auth_token"] = str((existing or {}).get("auth_token") or "")
        out.append(clone)
    return out


_IMMEDIATE_KEYS = frozenset({
    "TOTAL_BUDGET",
    "OUROBOROS_SOFT_TIMEOUT_SEC",
    "OUROBOROS_HARD_TIMEOUT_SEC",
    "OUROBOROS_TOOL_TIMEOUT_SEC",
    "GITHUB_TOKEN",
    "GITHUB_REPO",
})

_RESTART_REQUIRED_KEYS = frozenset({
    "OUROBOROS_MAX_WORKERS",
    "OUROBOROS_SERVER_HOST",
    "LOCAL_MODEL_SOURCE",
    "LOCAL_MODEL_FILENAME",
    "LOCAL_MODEL_PORT",
    "LOCAL_MODEL_N_GPU_LAYERS",
    "LOCAL_MODEL_CONTEXT_LENGTH",
    "LOCAL_MODEL_CHAT_FORMAT",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL",
    "GIGACHAT_SCOPE",
    "GIGACHAT_BASE_URL",
    "GIGACHAT_VERIFY_SSL_CERTS",
})


def _classify_settings_changes(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> list:
    """Return changed keys requiring process restart; others hot-reload next task."""
    return [
        k for k in _RESTART_REQUIRED_KEYS
        if str(new.get(k, "") or "") != str(old.get(k, "") or "")
    ]


def _merge_settings_payload(current: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    merged = {k: v for k, v in current.items()}
    for key in _SETTINGS_DEFAULTS:
        # Runtime mode is owner-only; loopback HTTP settings cannot raise scope.
        if key in {"OUROBOROS_RUNTIME_MODE", "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"}:
            continue
        if key not in body:
            continue
        if key in _SECRET_SETTING_KEYS and _looks_masked_secret(body[key]) and merged.get(key):
            continue
        merged[key] = body[key]
    for key, value in body.items():
        text_key = str(key or "").strip().upper()
        if text_key in _SETTINGS_DEFAULTS or text_key == "OUROBOROS_RUNTIME_MODE":
            continue
        if not _CUSTOM_SECRET_KEY_RE.match(text_key):
            continue
        if text_key.startswith("OUROBOROS_"):
            continue
        if _looks_masked_secret(value) and merged.get(text_key):
            continue
        merged[text_key] = value
    return merged


def _current_bind_host(request: Request) -> str:
    return str(getattr(getattr(request.app, "state", None), "bind_host", "") or "")


def _port_file(request: Request) -> pathlib.Path:
    configured = getattr(getattr(request.app, "state", None), "port_file", None)
    return pathlib.Path(configured) if configured is not None else pathlib.Path(DATA_DIR) / "state" / "server_port"


def _default_port(request: Request) -> int:
    return int(getattr(getattr(request.app, "state", None), "default_port", DEFAULT_PORT) or DEFAULT_PORT)


def _start_supervisor_if_needed_for_request(request: Request, settings: dict) -> bool:
    callback = getattr(getattr(request.app, "state", None), "start_supervisor_if_needed", None)
    return bool(callback(settings)) if callable(callback) else False


def _owner_audit(request: Request, action: str, payload: Dict[str, Any]) -> None:
    try:
        drive_root = request_drive_root(request)
    except Exception:
        drive_root = pathlib.Path(DATA_DIR)
    try:
        client = getattr(request, "client", None)
        append_jsonl(
            drive_root / "logs" / "events.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "owner_api_action",
                "action": str(action or ""),
                "client_host": str(getattr(client, "host", "") or ""),
                **{
                    key: value
                    for key, value in dict(payload or {}).items()
                    if "key" not in str(key).lower() and "secret" not in str(key).lower()
                },
            },
        )
    except Exception:
        log.debug("Failed to write owner API audit event", exc_info=True)


def _owner_write_settings(settings: Dict[str, Any]) -> None:
    """Write owner-controlled settings without applying the runtime-mode ratchet."""
    from ouroboros import config as _config

    _config._guard_live_settings_write()
    _config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd = _config._acquire_settings_lock()
    try:
        atomic_write_json(_config.SETTINGS_PATH, dict(settings), trailing_newline=False)
    finally:
        _config._release_settings_lock(fd)


def _owner_read_settings_raw() -> Dict[str, Any]:
    """Read settings for owner endpoints without applying runtime-mode ratchets."""
    from ouroboros import config as _config

    merged = dict(_SETTINGS_DEFAULTS)
    try:
        if _config.SETTINGS_PATH.exists():
            raw = json.loads(_config.SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged.update(raw)
    except Exception:
        log.debug("Failed to read raw owner settings; using defaults", exc_info=True)
    return merged


async def api_owner_runtime_mode(request: Request) -> JSONResponse:
    """Persist the owner-selected runtime mode for the next boot."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    from ouroboros import config as _config

    raw_mode = str((body or {}).get("mode") or "").strip().lower()
    if raw_mode not in set(_config.VALID_RUNTIME_MODES):
        return json_error("'mode' must be one of: light, advanced, pro", 400)
    old_settings = _owner_read_settings_raw()
    previous_mode = _config.normalize_runtime_mode(old_settings.get("OUROBOROS_RUNTIME_MODE"))
    active_mode = _config.get_runtime_mode()
    next_mode = _config.normalize_runtime_mode(raw_mode)
    restart_required = active_mode != next_mode
    current = dict(old_settings)
    current["OUROBOROS_RUNTIME_MODE"] = next_mode
    _owner_write_settings(current)
    _owner_audit(
        request,
        "runtime_mode",
        {
            "runtime_mode": next_mode,
            "previous_runtime_mode": previous_mode,
            "active_runtime_mode": active_mode,
            "restart_required": restart_required,
        },
    )
    return JSONResponse({
        "ok": True,
        "runtime_mode": next_mode,
        "restart_required": restart_required,
    })


async def api_owner_auto_grant(request: Request) -> JSONResponse:
    """Persist the owner auto-grant toggle outside generic settings writes."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict) or not isinstance(body.get("enabled"), bool):
        return json_error("'enabled' must be a boolean", 400)
    enabled = bool(body.get("enabled"))
    current = _owner_read_settings_raw()
    current["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = "true" if enabled else "false"
    _owner_write_settings(current)
    os.environ["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = current["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"]
    _owner_audit(request, "auto_grant", {"enabled": enabled})
    return JSONResponse({"ok": True, "enabled": enabled})

def _claude_code_status_payload() -> Dict[str, Any]:
    """Return app-managed Claude runtime status, versions, readiness, and stderr."""
    from ouroboros.platform_layer import resolve_claude_runtime

    rt = resolve_claude_runtime()
    label = rt.status_label()

    stderr_tail = ""
    try:
        from ouroboros.gateways.claude_code import get_last_stderr as gw_stderr
        stderr_tail = gw_stderr(max_chars=2000)
    except Exception:
        pass

    message_map = {
        "ready": f"Claude runtime ready (SDK {rt.sdk_version}, CLI {rt.cli_version})",
        "no_api_key": f"Claude runtime available (SDK {rt.sdk_version}) but ANTHROPIC_API_KEY is not set. Add it in Settings.",
        "error": f"Claude runtime error: {rt.error}",
        "degraded": f"Claude runtime degraded (SDK {rt.sdk_version}, CLI {'found' if rt.cli_path else 'missing'}). Try Repair.",
        "missing": "Claude runtime not available. Use Repair in Settings or reinstall the app.",
    }

    return {
        "status": label,
        "installed": bool(rt.sdk_version),
        "ready": rt.ready,
        "busy": False,
        "version": rt.sdk_version,
        "cli_version": rt.cli_version,
        "cli_path": rt.cli_path,
        "interpreter_path": rt.interpreter_path,
        "app_managed": rt.app_managed,
        "legacy_detected": rt.legacy_detected,
        "legacy_sdk_version": rt.legacy_sdk_version,
        "api_key_set": rt.api_key_set,
        "message": message_map.get(label, f"Claude runtime: {label}"),
        "error": rt.error,
        "stderr_tail": stderr_tail,
    }


async def api_settings_get(request: Request) -> JSONResponse:
    settings, _, _ = apply_runtime_provider_defaults(load_settings())
    safe = {k: v for k, v in settings.items()}
    for key in _SECRET_SETTING_KEYS:
        if safe.get(key):
            safe[key] = _mask_secret_value(safe[key])
    safe["MCP_SERVERS"] = _mask_mcp_servers_payload(safe.get("MCP_SERVERS") or [])
    for key, value in list(safe.items()):
        if key in _SECRET_SETTING_KEYS or key in _SETTINGS_DEFAULTS:
            continue
        if _CUSTOM_SECRET_KEY_RE.match(str(key)) and value:
            safe[key] = _mask_secret_value(value)
    try:
        port = int(_port_file(request).read_text().strip()) if _port_file(request).exists() else _default_port(request)
    except (ValueError, OSError):
        port = _default_port(request)
    meta = _build_network_meta(_current_bind_host(request), port)
    meta["custom_secret_keys"] = sorted(
        key for key in settings
        if key not in _SECRET_SETTING_KEYS
        and key not in _SETTINGS_DEFAULTS
        and _CUSTOM_SECRET_KEY_RE.match(str(key))
        and settings.get(key)
    )
    meta["setup_contract"] = build_setup_contract("web")
    safe["_meta"] = meta
    return JSONResponse(safe)


async def api_onboarding(request: Request) -> Response:
    settings, provider_defaults_changed, _provider_default_keys = apply_runtime_provider_defaults(load_settings())
    if provider_defaults_changed:
        save_settings(settings, allow_elevation=True)
    if has_startup_ready_provider(settings):
        return Response(status_code=204)
    return HTMLResponse(build_onboarding_html(settings, host_mode="web"))


async def api_claude_code_status(request: Request) -> JSONResponse:
    try:
        payload = await asyncio.to_thread(_claude_code_status_payload)
        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "installed": False,
            "busy": False,
            "message": "Failed to read Claude Agent SDK status.",
            "error": str(e),
        }, status_code=500)


async def api_claude_code_install(request: Request) -> JSONResponse:
    """Repair/update Claude runtime using the app-managed interpreter."""
    try:
        import subprocess as _sp
        import sys as _sys

        interpreter = _sys.executable
        try:
            from ouroboros.platform_layer import resolve_claude_runtime
            rt = resolve_claude_runtime()
            if rt.interpreter_path:
                interpreter = rt.interpreter_path
        except Exception:
            pass

        # Import SDK baseline at call time: one SSOT, clean endpoint error if broken.
        from ouroboros.launcher_bootstrap import _CLAUDE_SDK_BASELINE as sdk_baseline

        result = await asyncio.to_thread(
            lambda: _sp.run(
                [interpreter, "-m", "pip", "install", "--upgrade", sdk_baseline],
                capture_output=True, text=True, timeout=120,
            )
        )
        if result.returncode == 0:
            payload = await asyncio.to_thread(_claude_code_status_payload)
            payload["repaired"] = True
            return JSONResponse(payload)
        return JSONResponse({
            "status": "error",
            "installed": False,
            "ready": False,
            "busy": False,
            "message": "Claude runtime repair failed.",
            "error": (result.stderr or result.stdout or "")[:500],
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "installed": False,
            "ready": False,
            "busy": False,
            "message": "Claude runtime repair failed.",
            "error": f"{type(e).__name__}: {e}",
        }, status_code=500)


async def api_settings_post(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            return json_error("JSON body must be an object.", 400)
        parsed_budget: dict[str, float] = {}
        for budget_key in BUDGET_SETTING_KEYS:
            if budget_key not in body:
                continue
            budget_value, budget_error = parse_budget_setting(budget_key, body.get(budget_key))
            if budget_error:
                return json_error(budget_error, 400)
            if budget_value is not None:
                parsed_budget[budget_key] = budget_value
        if parsed_budget:
            body = dict(body)
            body.update(parsed_budget)
        old_settings = load_settings()
        from ouroboros.config import get_runtime_mode, normalize_runtime_mode as _norm_runtime_mode

        raw_old_settings = _owner_read_settings_raw()
        pending_runtime_mode = _norm_runtime_mode(
            raw_old_settings.get("OUROBOROS_RUNTIME_MODE", old_settings.get("OUROBOROS_RUNTIME_MODE"))
        )
        current_runtime_mode = get_runtime_mode()
        old_effective_settings = dict(old_settings)
        old_effective_settings["OUROBOROS_RUNTIME_MODE"] = current_runtime_mode
        if "MCP_SERVERS" in body:
            body = dict(body)
            body["MCP_SERVERS"] = _rehydrate_mcp_servers_payload(
                body.get("MCP_SERVERS"),
                old_settings.get("MCP_SERVERS"),
            )
        current = _merge_settings_payload(old_effective_settings, body)
        # Generic settings saves operate on the current boot baseline. A pending
        # next-boot mode written by /api/owner/runtime-mode is preserved on disk
        # below, but never hot-applied to this process/env.
        current["OUROBOROS_RUNTIME_MODE"] = current_runtime_mode
        # Trim opaque path text so configured/empty state is deterministic.
        current["OUROBOROS_SKILLS_REPO_PATH"] = str(
            current.get("OUROBOROS_SKILLS_REPO_PATH") or ""
        ).strip()
        try:
            from ouroboros.server_auth import is_loopback_host
            desired_host = str(current.get("OUROBOROS_SERVER_HOST") or "").strip()
            desired_password = str(current.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
            trust_unauth = _trust_nonlocal_bind_without_password_enabled()
            allowed_saved_hosts = {"", "127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0", "::", "[::]"}
            if desired_host and desired_host not in allowed_saved_hosts:
                return json_error(
                    "Server Bind Host in Settings supports localhost or wildcard "
                    "binds only (127.0.0.1 or 0.0.0.0). Specific LAN IP binds "
                    "are manual/env-only so the desktop launcher can keep using "
                    "a reliable loopback health check.",
                    400,
                )
            if desired_host and not is_loopback_host(desired_host) and not desired_password and not trust_unauth:
                return json_error(
                    "Setting a non-localhost Server Bind Host through the web UI "
                    "requires a Network Password in the same save. For manual "
                    "trusted-lab/Docker setups, stop Ouroboros and edit "
                    "settings.json or environment variables directly.",
                    400,
                )
            current_effective_host = (
                str(_current_bind_host(request) or "").strip()
                or str(os.environ.get("OUROBOROS_SERVER_HOST") or "").strip()
            )
            old_password = str(old_settings.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
            if (
                current_effective_host
                and not is_loopback_host(current_effective_host)
                and old_password
                and not desired_password
                and not trust_unauth
            ):
                return json_error(
                    "Cannot clear Network Password while the running server is "
                    "still bound to a non-localhost interface. First save a "
                    "loopback Server Bind Host and restart, then clear the password.",
                    400,
                )
        except Exception:
            log.warning("Could not validate network bind settings", exc_info=True)
        current, provider_defaults_changed, provider_default_keys = apply_runtime_provider_defaults(current)
        if str(current.get("LOCAL_MODEL_SOURCE", "") or "").strip() and not has_supervisor_provider(current):
            return json_error("Local-only setups must route at least one model to the local runtime.", 400)
        all_changed = [
            k for k in current
            if str(current.get(k, "") or "") != str(old_effective_settings.get(k, "") or "")
        ]
        restart_keys = _classify_settings_changes(old_effective_settings, current)

        settings_to_save = dict(current)
        settings_to_save["OUROBOROS_RUNTIME_MODE"] = pending_runtime_mode
        _owner_write_settings(settings_to_save)
        _apply_settings_to_env(current)
        _start_supervisor_if_needed_for_request(request, current)

        if any(k in all_changed for k in ("MCP_ENABLED", "MCP_SERVERS", "MCP_TOOL_TIMEOUT_SEC")):
            try:
                from ouroboros.mcp_client import (
                    reconfigure_from_settings as _mcp_reconfigure,
                    refresh_all_background as _mcp_refresh_background,
                )
                _mcp_reconfigure(current)
                _mcp_refresh_background(reason="settings")
            except Exception:
                log.warning("MCP reconfigure after settings change failed", exc_info=True)

        # Skills repo/runtime changes require extension loader reconciliation.
        try:
            from ouroboros.extension_loader import reload_all as _reload_extensions
            new_path = str(current.get("OUROBOROS_SKILLS_REPO_PATH") or "").strip()
            old_path = str(old_effective_settings.get("OUROBOROS_SKILLS_REPO_PATH") or "").strip()
            new_runtime_mode = str(current.get("OUROBOROS_RUNTIME_MODE") or "").strip()
            old_runtime_mode = str(old_effective_settings.get("OUROBOROS_RUNTIME_MODE") or "").strip()
            if new_path != old_path or new_runtime_mode != old_runtime_mode:
                # Use load_settings so extensions do not capture a stale snapshot.
                from ouroboros.config import load_settings as _load_settings
                reload_drive_root = pathlib.Path(
                    request.app.state.drive_root
                    if hasattr(request.app, "state") and hasattr(request.app.state, "drive_root")
                    else request_drive_root(request)
                )
                if (
                    (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules)
                    and reload_drive_root == pathlib.Path.home() / "Ouroboros" / "data"
                    and not os.environ.get("OUROBOROS_DATA_DIR")
                ):
                    log.info("Skipping extension reload_all against real DATA_DIR during pytest settings save")
                else:
                    _reload_extensions(
                        reload_drive_root,
                        _load_settings,
                        repo_path=new_path or None,
                    )
        except Exception:
            log.error("Extension reload after settings change failed", exc_info=True)

        try:
            from supervisor.state import refresh_budget_from_settings
            refresh_budget_from_settings(current)
        except Exception:
            pass
        try:
            from supervisor.queue import refresh_timeouts_from_settings
            refresh_timeouts_from_settings(current)
        except Exception:
            pass
        try:
            from supervisor.message_bus import refresh_budget_limit
            raw_budget = current.get("TOTAL_BUDGET")
            new_budget = float(raw_budget) if raw_budget is not None else 0.0
            refresh_budget_limit(new_budget)
        except Exception:
            pass

        warnings = []
        if provider_defaults_changed:
            change_kind = classify_runtime_provider_change(old_effective_settings, current)
            if change_kind == "direct_normalize":
                warnings.append(
                    "Normalized direct-provider routing because OpenRouter is not configured for the active provider."
                )
        try:
            from supervisor.message_bus import get_bridge
            get_bridge().configure_from_settings(current)
        except Exception:
            pass
        try:
            from ouroboros.server_auth import is_loopback_host
            desired_host = str(current.get("OUROBOROS_SERVER_HOST") or "").strip()
            desired_password = str(current.get("OUROBOROS_NETWORK_PASSWORD") or "").strip()
            if desired_host and not is_loopback_host(desired_host) and not desired_password:
                if _trust_nonlocal_bind_without_password_enabled():
                    warnings.append(
                        "OUROBOROS_TRUST_NONLOCAL_BIND_WITHOUT_PASSWORD=1 allows this "
                        "non-localhost bind without Ouroboros's internal Network Password. "
                        "Use only behind ingress auth, VPN, private networking, or an auth proxy."
                    )
                else:
                    warnings.append(
                        "Server Bind Host is non-localhost and Network Password is empty; "
                        "after restart the app will be reachable on the network without a password."
                    )
        except Exception:
            pass
        _repo_slug = current.get("GITHUB_REPO", "")
        _gh_token = current.get("GITHUB_TOKEN", "")
        if _gh_token and any(k in all_changed for k in ("GITHUB_REPO", "GITHUB_TOKEN")):
            from supervisor.git_ops import configure_personal_remote
            remote_ok, remote_msg, resolved_slug = configure_personal_remote(
                _repo_slug,
                _gh_token,
                auto_fork=not bool(str(_repo_slug or "").strip()),
                confirm_replace_origin=bool(body.get("GITHUB_REPLACE_ORIGIN_CONFIRMED")),
            )
            if not remote_ok:
                log.warning("Remote configuration failed on settings save: %s", remote_msg)
                warnings.append(f"Remote config failed: {remote_msg}")
            elif resolved_slug and resolved_slug != _repo_slug:
                current["GITHUB_REPO"] = resolved_slug
                settings_to_save["GITHUB_REPO"] = resolved_slug
                _owner_write_settings(settings_to_save)
                os.environ["GITHUB_REPO"] = resolved_slug
        immediate_changed = [k for k in all_changed if k in _IMMEDIATE_KEYS]
        next_task_changed = [
            k for k in all_changed
            if k not in _IMMEDIATE_KEYS and k not in _RESTART_REQUIRED_KEYS
        ]
        resp: Dict[str, Any] = {"status": "saved"}
        if not all_changed:
            resp["no_changes"] = True
        if restart_keys:
            resp["restart_required"] = True
            resp["restart_keys"] = restart_keys
        if immediate_changed:
            resp["immediate_changed"] = True
        if next_task_changed:
            resp["next_task_changed"] = True
        if warnings:
            resp["warnings"] = warnings
        return JSONResponse(resp)
    except Exception as e:
        return json_exception(e, 400)
