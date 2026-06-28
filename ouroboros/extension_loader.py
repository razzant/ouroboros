"""Load reviewed in-process ``type: extension`` skills through PluginAPI.

Extensions run inside Ouroboros, so imports are allowed only after a fresh
executable skill review, manifest permissions, and owner grants pass. All
registered surfaces are provider-safe namespaced and tracked per skill so
disable/reload can tear them down and purge modules cleanly.
"""

from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
import hashlib
import json
import logging
import os
import pathlib
import re
import secrets
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence

from ouroboros.contracts.plugin_api import (
    ExtensionRegistrationError,
    ExecutionMode,
    FORBIDDEN_EXTENSION_SETTINGS,
    VALID_EXTENSION_PERMISSIONS,
    VALID_EXTENSION_ROUTE_METHODS,
    available_capabilities,
    capability_available,
)
from ouroboros.event_bus import get_global_event_bus
from ouroboros.extension_companion import CompanionDescriptor, get_global_supervisor, is_server_process
from ouroboros.extension_ui_validation import _assert_ws_message_type, validate_ui_render as _validate_ui_render
from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME
from ouroboros.extension_isolated_deps import _isolated_python_site_dirs, async_isolated_site_dirs_scope, isolated_site_dirs_scope, is_skill_cache_path
from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES, LoadedSkill, SkillPayloadUnreadable, compute_content_hash, discover_skills, find_skill, grant_status_for_skill, requested_core_setting_keys, skill_review_gate, skill_state_dir
from ouroboros.skill_token import SkillToken
from ouroboros.tools.skill_exec import _scrub_env
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


# Registration bookkeeping.


@dataclass
class _ExtensionRegistrations:
    """Attached surfaces owned by one loaded extension."""

    tools: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    ws_handlers: List[str] = field(default_factory=list)
    ui_tabs: List[str] = field(default_factory=list)
    settings_sections: List[str] = field(default_factory=list)
    unload_callbacks: List[Callable[[], Any]] = field(default_factory=list)
    event_subscriptions: List[str] = field(default_factory=list)
    companion_names: List[str] = field(default_factory=list)
    supervised_futures: List[Any] = field(default_factory=list)
    api_instances: List[Any] = field(default_factory=list)
    content_hash: Optional[str] = None
    skill_dir: Optional[str] = None
    import_root: Optional[str] = None


@dataclass
class _ExtensionLoadFailure:
    content_hash: str
    skill_dir: str
    error: str


@dataclass
class _PluginAPIConfig:
    skill_name: str
    permissions: Sequence[str]
    env_allowlist: Sequence[str]
    state_dir: pathlib.Path
    settings_reader: Callable[[], Dict[str, Any]]
    granted_keys: Sequence[str] | None = None
    subscribe_events: Sequence[str] | None = None
    companion_processes: Sequence[Dict[str, Any]] | None = None
    skill_dir: pathlib.Path | None = None
    runtime_skill_dir: pathlib.Path | None = None
    dependency_site_dirs_enabled: bool = False


# Lock-guarded registries; per-surface maps keep unload proportional to one extension.
_lock = threading.RLock()
_extensions: Dict[str, _ExtensionRegistrations] = {}
_extension_modules: Dict[str, ModuleType] = {}
_load_failures: Dict[str, _ExtensionLoadFailure] = {}
_unloading: set[str] = set()
_lifecycle_locks: Dict[str, threading.RLock] = {}
_tools: Dict[str, Any] = {}            # {"ext_<len>_<token>_<name>": ToolEntry-like}
_routes: Dict[str, Any] = {}           # {"/api/extensions/<skill>/<path>": handler_spec}
_ws_handlers: Dict[str, Any] = {}      # {"ext_<len>_<token>_<message_type>": handler}
_ui_tabs: Dict[str, Any] = {}          # {"<skill>:<tab_id>": tab_spec}
# Declarative settings sections keyed like UI tabs.
_settings_sections: Dict[str, Any] = {}
_ws_broadcaster: Optional[Callable[[dict], None]] = None
_EXTENSION_NAME_PREFIX = "ext_"
_EXTENSION_SKILL_TOKEN_MAX = 32
_EXTENSION_SHORT_MAX = 24
_EXTENSION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _out_of_process_handler_proxy(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("extension surface is configured for out-of-process dispatch")


def current_execution_mode() -> ExecutionMode:
    """Execution context of the running PluginAPI, derived from the child env flag."""
    if os.environ.get("OUROBOROS_EXTENSION_PROCESS_CHILD") == "1":
        return ExecutionMode.OUT_OF_PROCESS
    return ExecutionMode.IN_PROCESS


def _record_companion_name(bundle: _ExtensionRegistrations, name: str) -> None:
    if name not in bundle.companion_names:
        bundle.companion_names.append(name)


def _request_server_reconcile_if_worker(
    drive_root: pathlib.Path | None,
    skill_name: str,
    *,
    reason: str,
) -> None:
    """Signal the server process after a worker-side extension state change."""
    if drive_root is None or is_server_process():
        return
    try:
        from ouroboros.extension_reconcile_queue import request_extension_reconcile

        request_extension_reconcile(drive_root, skill_name, reason=reason, source="worker")
    except Exception:
        log.debug("Failed to request server extension reconcile for %s", skill_name, exc_info=True)


def _reject_extension_child_side_effect(capability: str) -> None:
    """Enforce the contract capability matrix for the current execution mode.

    Every side-effect registration method calls this; the matrix in
    ``contracts.plugin_api`` is the single source of truth for what an
    out-of-process (isolated-dep) child may use. on_unload, send_ws_message, and
    register_companion_process are supported out-of-process; subscribe_event and
    register_supervised_task are not (use a companion_process instead).
    """

    mode = current_execution_mode()
    if not capability_available(capability, mode):
        available = ", ".join(sorted(available_capabilities(mode)))
        raise ExtensionRegistrationError(
            f"{capability} is not available to out-of-process (isolated-dep) extensions "
            f"in the per-call child; declare a companion_process for long-running work "
            f"and host-event subscription. Available capabilities here: {available}."
        )


def _validate_child_catalog_namespace(skill_name: str, surface_kind: str, value: str) -> None:
    """Re-check child catalog namespaces at the host trust boundary."""

    if surface_kind in {"tool", "ws handler"}:
        expected = extension_name_prefix(skill_name)
    elif surface_kind == "route":
        expected = f"/api/extensions/{skill_name}/"
    elif surface_kind in {"ui tab", "settings section"}:
        expected = f"{skill_name}:"
    else:
        expected = ""
    if expected and not value.startswith(expected):
        raise ExtensionRegistrationError(
            f"out-of-process {surface_kind} {value!r} escaped extension namespace {expected!r}"
        )


def _validate_child_tool_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    name = str(item.get("name") or "")
    _validate_child_catalog_namespace(skill_name, "tool", name)
    if not _EXTENSION_NAME_RE.match(name):
        raise ExtensionRegistrationError(f"out-of-process tool {name!r} is not provider-safe")
    if not isinstance(item.get("schema", {}), dict):
        raise ExtensionRegistrationError(f"out-of-process tool {name!r} schema must be an object")
    item["schema"] = dict(item.get("schema") or {})
    item["description"] = str(item.get("description") or "")
    try:
        item["timeout_sec"] = max(1, int(item.get("timeout_sec") or 60))
    except (TypeError, ValueError) as exc:
        raise ExtensionRegistrationError(f"out-of-process tool {name!r} timeout_sec must be an integer") from exc
    return item


def _validate_child_route_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    path = str(item.get("path") or "")
    _validate_child_catalog_namespace(skill_name, "route", path)
    methods_iter = item.get("methods") or ("GET",)
    if isinstance(methods_iter, str):
        methods_iter = (methods_iter,)
    methods = tuple(dict.fromkeys(str(method).strip().upper() for method in methods_iter if str(method).strip()))
    if not methods:
        raise ExtensionRegistrationError(f"out-of-process route {path!r} methods must be non-empty")
    invalid = [method for method in methods if method not in VALID_EXTENSION_ROUTE_METHODS]
    if invalid:
        raise ExtensionRegistrationError(
            f"out-of-process route {path!r} methods {invalid!r} are unsupported; "
            f"expected subset of {sorted(VALID_EXTENSION_ROUTE_METHODS)}"
        )
    item["methods"] = methods
    return item


def _validate_child_ws_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    msg_type = str(item.get("type") or "")
    _validate_child_catalog_namespace(skill_name, "ws handler", msg_type)
    if not _EXTENSION_NAME_RE.match(msg_type):
        raise ExtensionRegistrationError(f"out-of-process ws handler {msg_type!r} is not provider-safe")
    return item


def _validate_child_ui_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    key = str(item.get("key") or "")
    _validate_child_catalog_namespace(skill_name, "ui tab", key)
    if not isinstance(item.get("render", {}), dict):
        raise ExtensionRegistrationError(f"out-of-process ui tab {key!r} render must be an object")
    render = _validate_ui_render(dict(item.get("render") or {}))
    item["render"] = render
    span = _widget_span_from_render(render)
    item["span"] = span
    item["grid_span"] = span
    return item


def _validate_child_settings_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    key = str(item.get("key") or "")
    _validate_child_catalog_namespace(skill_name, "settings section", key)
    if not isinstance(item.get("render", {}), dict):
        raise ExtensionRegistrationError(f"out-of-process settings section {key!r} render must be an object")
    item["render"] = _validate_ui_render(dict(item.get("render") or {}))
    return item


def _register_out_of_process_surfaces(
    skill: LoadedSkill,
    *,
    current_hash: str,
    catalog: Dict[str, Any],
) -> None:
    """Install proxy surface descriptors returned by a child catalog run."""

    with _lock:
        bundle = _extensions.get(skill.name)
        if bundle is None:
            bundle = _ExtensionRegistrations()
            _extensions[skill.name] = bundle
        bundle.content_hash = current_hash
        bundle.skill_dir = str(skill.skill_dir.resolve())
        bundle.import_root = None
        _load_failures.pop(skill.name, None)

        for raw in catalog.get("tools") or []:
            item = _validate_child_tool_descriptor(skill.name, dict(raw or {}))
            name = str(item.get("name") or "")
            if not name:
                continue
            if name in _tools:
                raise ExtensionRegistrationError(f"tool {name!r} already registered")
            item["handler"] = _out_of_process_handler_proxy
            item["skill"] = skill.name
            item["out_of_process"] = True
            item["skills_repo_path"] = str(skill.skill_dir.parent)
            _tools[name] = item
            bundle.tools.append(name)

        for raw in catalog.get("routes") or []:
            item = _validate_child_route_descriptor(skill.name, dict(raw or {}))
            path = str(item.get("path") or "")
            if not path:
                continue
            if path in _routes:
                raise ExtensionRegistrationError(f"route {path!r} already registered")
            item["handler"] = _out_of_process_handler_proxy
            item["skill"] = skill.name
            item["out_of_process"] = True
            item["skills_repo_path"] = str(skill.skill_dir.parent)
            _routes[path] = item
            bundle.routes.append(path)

        for raw in catalog.get("ws_handlers") or []:
            item = _validate_child_ws_descriptor(skill.name, dict(raw or {}))
            msg_type = str(item.get("type") or "")
            if not msg_type:
                continue
            if msg_type in _ws_handlers:
                raise ExtensionRegistrationError(f"ws handler {msg_type!r} already registered")
            item["handler"] = _out_of_process_handler_proxy
            item["skill"] = skill.name
            item["out_of_process"] = True
            item["skills_repo_path"] = str(skill.skill_dir.parent)
            _ws_handlers[msg_type] = item
            bundle.ws_handlers.append(msg_type)

        for raw in catalog.get("ui_tabs") or []:
            item = _validate_child_ui_descriptor(skill.name, dict(raw or {}))
            key = str(item.pop("key", "") or "")
            if not key:
                continue
            if key in _ui_tabs:
                raise ExtensionRegistrationError(f"ui tab {key!r} already registered")
            _ui_tabs[key] = item
            bundle.ui_tabs.append(key)

        for raw in catalog.get("settings_sections") or []:
            item = _validate_child_settings_descriptor(skill.name, dict(raw or {}))
            key = str(item.pop("key", "") or "")
            if not key:
                continue
            if key in _settings_sections:
                raise ExtensionRegistrationError(f"settings section {key!r} already registered")
            _settings_sections[key] = item
            bundle.settings_sections.append(key)


def _spawn_out_of_process_companions(
    skill: LoadedSkill,
    *,
    catalog: Dict[str, Any],
    state_dir: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    granted_keys: Sequence[str],
    dependency_site_dirs_enabled: bool,
) -> None:
    """Host-spawn companions an isolated-dep extension declared during catalog.

    Reuses the in-process ``register_companion_process`` path (the host is the
    server process, so it owns the supervisor) instead of duplicating descriptor
    construction. Cataloged names are re-validated against the reviewed manifest at
    the host trust boundary before any process is started.
    """

    names = [str(n).strip() for n in (catalog.get("companions") or []) if str(n).strip()]
    if not names:
        return
    declared = {
        str(item.get("name") or "").strip()
        for item in (skill.manifest.companion_processes or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill.name,
        permissions=list(skill.manifest.permissions or []),
        env_allowlist=list(skill.manifest.env_from_settings or []),
        state_dir=state_dir,
        settings_reader=settings_reader,
        granted_keys=list(granted_keys),
        subscribe_events=list(getattr(skill.manifest, "subscribe_events", []) or []),
        companion_processes=list(getattr(skill.manifest, "companion_processes", []) or []),
        skill_dir=skill.skill_dir,
        runtime_skill_dir=skill.skill_dir,
        dependency_site_dirs_enabled=dependency_site_dirs_enabled,
    ))
    for name in names:
        if name not in declared:
            raise ExtensionRegistrationError(
                f"out-of-process companion {name!r} escaped manifest.companion_processes"
            )
        api.register_companion_process(name)


def mint_skill_token(state_dir: pathlib.Path, skill_name: str, skill_dir: Optional[pathlib.Path]) -> str:
    """Read or rotate the per-skill Host Service token, bound to the content hash.

    Shared by the in-process PluginAPI (``get_skill_token``) and the out-of-process
    child env builder so a child/companion can authenticate to the Host Service.
    """
    token_path = pathlib.Path(state_dir) / AUTH_TOKEN_FILENAME
    payload = read_json_dict(token_path) or {}
    token = str(payload.get("token") or "")
    content_hash = ""
    if skill_dir is not None:
        try:
            content_hash = compute_content_hash(pathlib.Path(skill_dir))
        except Exception:
            content_hash = ""
    if not token or str(payload.get("content_hash") or "") != content_hash:
        token = secrets.token_urlsafe(32)
        atomic_write_json(
            token_path,
            {
                "token": token,
                "issued_at": utc_now_iso(),
                "skill": skill_name,
                "content_hash": content_hash,
            },
        )
        try:
            token_path.chmod(0o600)
        except OSError:
            log.debug("Failed to chmod skill token file %s", token_path, exc_info=True)
    return token


def _extension_skill_token(skill_name: str) -> str:
    """Return a short ASCII token without changing skill identity."""
    text = str(skill_name or "").strip()
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "_" for ch in text)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    raw_budget = _EXTENSION_SKILL_TOKEN_MAX - 2
    if safe and safe == text and len(safe) <= raw_budget:
        return f"r_{safe}"
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:10]
    prefix_budget = _EXTENSION_SKILL_TOKEN_MAX - len(digest) - 3
    prefix = (safe or "skill")[:prefix_budget].strip("_-") or "skill"
    return f"h_{prefix}_{digest}"


def extension_name_prefix(skill_name: str) -> str:
    """Return the provider-safe prefix for one extension."""
    token = _extension_skill_token(skill_name)
    return f"{_EXTENSION_NAME_PREFIX}{len(token)}_{token}_"


def extension_surface_name(skill_name: str, short_name: str) -> str:
    """Return a provider-safe canonical surface name."""
    full = f"{extension_name_prefix(skill_name)}{short_name}"
    if not _EXTENSION_NAME_RE.match(full):
        raise ExtensionRegistrationError(
            f"extension surface name {full!r} must match provider tool-name limits"
        )
    return full


def parse_extension_surface_name(name: str) -> tuple[str, str] | None:
    """Return ``(encoded_skill_token, short_name)`` for extension surface names."""
    text = str(name or "").strip()
    if not _EXTENSION_NAME_RE.match(text) or not text.startswith(_EXTENSION_NAME_PREFIX):
        return None
    rest = text[len(_EXTENSION_NAME_PREFIX):]
    length_text, sep, remainder = rest.partition("_")
    if sep != "_" or not length_text.isdigit():
        return None
    token_len = int(length_text)
    if token_len < 1 or len(remainder) <= token_len or remainder[token_len] != "_":
        return None
    token = remainder[:token_len]
    short = remainder[token_len + 1:]
    return token, short


def _lifecycle_lock_for(skill_name: str) -> threading.RLock:
    with _lock:
        lock = _lifecycle_locks.get(skill_name)
        if lock is None:
            lock = threading.RLock()
            _lifecycle_locks[skill_name] = lock
        return lock


def _run_unload_callback(skill_name: str, callback: Callable[[], Any], timeout_sec: float = 2.0) -> None:
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            callback()
        except BaseException as exc:  # pragma: no cover - surfaced via log
            errors.append(exc)

    thread = threading.Thread(target=runner, name=f"ouroboros-ext-unload-{skill_name}", daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        log.warning("extension %s unload callback timed out after %.1fs", skill_name, timeout_sec)
        return
    if errors:
        exc = errors[0]
        log.warning("extension %s unload callback failed", skill_name, exc_info=(type(exc), exc, exc.__traceback__))


# PluginAPI implementation.


def _assert_namespace_path(path: str) -> str:
    """Return a normalised relative path for route registration or raise."""
    rel = str(path or "").strip()
    if not rel:
        raise ExtensionRegistrationError("path must be non-empty")
    if rel.startswith("/"):
        raise ExtensionRegistrationError(
            f"path must be relative, not absolute: {rel!r}"
        )
    if ".." in pathlib.PurePosixPath(rel).parts:
        raise ExtensionRegistrationError(
            f"path must not contain '..' segments: {rel!r}"
        )
    return rel


def _assert_tool_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise ExtensionRegistrationError("tool name must be non-empty")
    if len(candidate) > _EXTENSION_SHORT_MAX:
        raise ExtensionRegistrationError(
            f"tool name must be <= {_EXTENSION_SHORT_MAX} characters: {candidate!r}"
        )
    if not candidate.replace("_", "").isalnum():
        raise ExtensionRegistrationError(
            f"tool name must be alnum/underscore only: {candidate!r}"
        )
    return candidate


def _widget_span_from_render(render: Dict[str, Any]) -> int:
    """Normalize optional UI-card width metadata from a render declaration."""
    raw = render.get("span", render.get("grid_span", 1))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return 2 if value >= 2 else 1


def set_ws_broadcaster(broadcaster: Callable[[dict], None] | None) -> None:
    """Install the host WebSocket broadcaster used by PluginAPI.send_ws_message."""
    global _ws_broadcaster
    with _lock:
        _ws_broadcaster = broadcaster


class PluginAPIImpl:
    """PluginAPI bound to one skill, permission set, and state dir."""

    def __init__(self, config: _PluginAPIConfig | None = None, **legacy: Any) -> None:
        if config is None:
            config = _PluginAPIConfig(**legacy)
        self._skill = config.skill_name
        self._permissions = frozenset(str(p).strip() for p in (config.permissions or []))
        self._env_allow = frozenset(str(k).strip() for k in (config.env_allowlist or []))
        self._env_allow_upper = frozenset(k.upper() for k in self._env_allow)
        self._state_dir = pathlib.Path(config.state_dir)
        self._subscribe_events = frozenset(str(t).strip() for t in (config.subscribe_events or []) if str(t).strip())
        self._companion_specs = {
            str(item.get("name") or "").strip(): dict(item)
            for item in (config.companion_processes or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        # Keep runtime_info cheap and tied to the loaded payload.
        self._skill_dir = pathlib.Path(config.skill_dir) if config.skill_dir is not None else None
        self._runtime_skill_dir = pathlib.Path(config.runtime_skill_dir) if config.runtime_skill_dir is not None else self._skill_dir
        self._dependency_site_dirs_enabled = bool(config.dependency_site_dirs_enabled)
        self._settings_reader = config.settings_reader
        self._registration_closed = False
        self._runtime_closing = False
        self._runtime_closed = False
        self._api_lock = threading.RLock()
        # Core settings are exposed only when a content-hash-bound owner grant
        # was already verified; otherwise the denylist silently drops them.
        self._granted_upper = frozenset(
            str(k).strip().upper() for k in (config.granted_keys or []) if str(k).strip()
        )

    # --- internal helpers ---

    def _require(self, perm: str) -> None:
        with _lock:
            self._require_open_locked()
        if perm not in VALID_EXTENSION_PERMISSIONS:
            raise ExtensionRegistrationError(
                f"unknown extension permission {perm!r}"
            )
        if perm not in self._permissions:
            raise ExtensionRegistrationError(
                f"skill {self._skill!r} cannot {perm!r} "
                f"— manifest permissions={sorted(self._permissions)}"
            )

    def _require_open_locked(self) -> None:
        if self._registration_closed or self._runtime_closing or self._runtime_closed or self._skill in _unloading:
            raise ExtensionRegistrationError(
                f"skill {self._skill!r} cannot register after unload has started"
            )

    def _wrap_runtime_handler(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        if self._skill_dir is None:
            return handler

        if inspect.iscoroutinefunction(handler):
            @functools.wraps(handler)
            async def _async_wrapped(*args: Any, **kwargs: Any) -> Any:
                async with async_isolated_site_dirs_scope(
                    self._skill_dir,
                    enabled=self._dependency_site_dirs_enabled,
                ):
                    return await handler(*args, **kwargs)

            return _async_wrapped

        @functools.wraps(handler)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            with isolated_site_dirs_scope(self._skill_dir, enabled=self._dependency_site_dirs_enabled):
                result = handler(*args, **kwargs)
                return result

        return _wrapped

    def _register_surface_locked(
        self,
        registry: Dict[str, Any],
        key: str,
        value: Dict[str, Any],
        bundle_attr: str,
        label: str,
    ) -> None:
        self._require_open_locked()
        if key in registry:
            raise ExtensionRegistrationError(f"{label} {key!r} already registered")
        registry[key] = value
        getattr(_extensions.setdefault(self._skill, _ExtensionRegistrations()), bundle_attr).append(key)

    # --- registration ---

    def register_tool(
        self,
        name: str,
        handler: Callable[..., str],
        *,
        description: str,
        schema: Dict[str, Any],
        timeout_sec: int = 60,
    ) -> None:
        self._require("tool")
        short = _assert_tool_name(name)
        full = extension_surface_name(self._skill, short)
        # Decide the ctx calling-convention on the RAW handler at register time:
        # the runtime wrapper is (*args, **kwargs), so inspecting it later always
        # reports VAR_POSITIONAL and forces a ctx-first call (TypeError for
        # keyword-only / zero-arg handlers). Dispatch reads this stored flag.
        from ouroboros.extension_process_runner import _handler_wants_ctx
        wants_ctx = _handler_wants_ctx(handler)
        with _lock:
            self._register_surface_locked(_tools, full, {
                "name": full,
                "handler": self._wrap_runtime_handler(handler),
                "wants_ctx": wants_ctx,
                "description": str(description or ""),
                "schema": dict(schema or {}),
                "timeout_sec": max(1, int(timeout_sec)),
                "skill": self._skill,
            }, "tools", "tool")

    def register_route(
        self,
        path: str,
        handler: Callable[..., Any],
        *,
        methods: Sequence[str] = ("GET",),
    ) -> None:
        self._require("route")
        rel = _assert_namespace_path(path)
        methods_iter = (methods,) if isinstance(methods, str) else (methods or ())
        norm_methods = tuple(
            dict.fromkeys(
                str(m).strip().upper()
                for m in methods_iter
                if str(m).strip()
            )
        )
        if not norm_methods:
            raise ExtensionRegistrationError("route methods must be non-empty")
        invalid_methods = [m for m in norm_methods if m not in VALID_EXTENSION_ROUTE_METHODS]
        if invalid_methods:
            raise ExtensionRegistrationError(
                f"route methods {invalid_methods!r} are unsupported; "
                f"expected subset of {sorted(VALID_EXTENSION_ROUTE_METHODS)}"
            )
        mount = f"/api/extensions/{self._skill}/{rel}"
        with _lock:
            self._register_surface_locked(_routes, mount, {
                "path": mount,
                "handler": self._wrap_runtime_handler(handler),
                "methods": norm_methods,
                "skill": self._skill,
            }, "routes", "route")

    def register_ws_handler(
        self,
        message_type: str,
        handler: Callable[..., Any],
    ) -> None:
        self._require("ws_handler")
        short = _assert_ws_message_type(message_type)
        full = extension_surface_name(self._skill, short)
        with _lock:
            self._register_surface_locked(_ws_handlers, full, {
                "type": full,
                "handler": self._wrap_runtime_handler(handler),
                "skill": self._skill,
            }, "ws_handlers", "ws handler")

    def register_ui_tab(
        self,
        tab_id: str,
        title: str,
        *,
        icon: str = "extension",
        render: Dict[str, Any] | None = None,
    ) -> None:
        self._require("widget")
        clean_tab = _assert_tool_name(tab_id)  # same syntax rules
        key = f"{self._skill}:{clean_tab}"
        validated_render = _validate_ui_render({} if render is None else render)
        span = _widget_span_from_render(validated_render)
        with _lock:
            self._register_surface_locked(_ui_tabs, key, {
                "skill": self._skill,
                "tab_id": clean_tab,
                "title": str(title or clean_tab),
                "icon": str(icon or "extension"),
                "ws_prefix": extension_name_prefix(self._skill),
                "render": validated_render,
                "span": span,
                "grid_span": span,
                "ui_host_pending": True,
            }, "ui_tabs", "ui tab")

    def register_settings_section(
        self,
        section_id: str,
        title: str,
        *,
        schema: Dict[str, Any],
    ) -> None:
        """Validate and register a declarative Settings UI section."""
        # Settings sections share the widget permission and host-rendered schema.
        self._require("widget")
        clean_id = _assert_tool_name(section_id)
        key = f"{self._skill}:{clean_id}"
        # Settings stay declarative-only and narrower than widgets.
        allowed = {"form", "action", "markdown", "json"}
        components = list((schema or {}).get("components") or [])
        for idx, component in enumerate(components):
            if not isinstance(component, dict):
                raise ExtensionRegistrationError(
                    f"settings section component {idx} must be an object"
                )
            ctype = str(component.get("type") or "").strip()
            if ctype not in allowed:
                raise ExtensionRegistrationError(
                    f"settings section component {idx} type {ctype!r} is unsupported; "
                    f"expected one of {sorted(allowed)}"
                )
        validated = _validate_ui_render({
            "kind": "declarative",
            "schema_version": 1,
            "components": components,
        })
        with _lock:
            self._register_surface_locked(_settings_sections, key, {
                "skill": self._skill,
                "section_id": clean_id,
                "title": str(title or clean_id),
                "render": validated,
            }, "settings_sections", "settings section")

    def register_supervised_task(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        restart_policy: str = "on_failure",
        max_restarts: int = 5,
        backoff_seconds: float = 2.0,
    ) -> None:
        """Declare a server-owned supervised task; workers only record it."""
        _reject_extension_child_side_effect("register_supervised_task")
        self._require("supervised_task")
        clean_name = _assert_tool_name(name)
        future = None
        if is_server_process():
            loop = getattr(get_global_event_bus(), "_loop", None)
            if loop is not None and loop.is_running():
                import asyncio

                async def _runner() -> None:
                    restarts = 0
                    while True:
                        try:
                            result = factory()
                            if inspect.isawaitable(result):
                                await result
                            return
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            restarts += 1
                            if restart_policy != "on_failure" or restarts > max_restarts:
                                log.warning("supervised task %s/%s stopped after failure", self._skill, clean_name, exc_info=True)
                                return
                            await asyncio.sleep(max(0.1, float(backoff_seconds)))

                future = asyncio.run_coroutine_threadsafe(_runner(), loop)
        with _lock:
            self._require_open_locked()
            bundle = _extensions.setdefault(self._skill, _ExtensionRegistrations())
            _record_companion_name(bundle, f"task:{clean_name}")
            if future is not None:
                bundle.supervised_futures.append(future)

    def register_companion_process(
        self,
        name: str,
    ) -> None:
        _reject_extension_child_side_effect("register_companion_process")
        self._require("companion_process")
        clean_name = _assert_tool_name(name)
        spec = self._companion_specs.get(clean_name)
        if spec is None:
            raise ExtensionRegistrationError(
                f"companion {clean_name!r} is not declared in manifest.companion_processes"
            )
        if current_execution_mode() is ExecutionMode.OUT_OF_PROCESS:
            # Catalog child: only record the manifest-declared name. The host spawns
            # and supervises the real companion after the catalog returns (it owns the
            # supervisor), reusing the in-process descriptor build below.
            with _lock:
                self._require_open_locked()
                bundle = _extensions.setdefault(self._skill, _ExtensionRegistrations())
                _record_companion_name(bundle, clean_name)
            return
        expected_cmd = [str(part) for part in (spec.get("command") or []) if str(part)]
        expected_runtime = str(spec.get("runtime") or "").strip()
        cmd = list(expected_cmd)
        if not cmd:
            raise ExtensionRegistrationError("companion command must be declared in manifest")
        if expected_runtime in {"python", "python3"} and cmd[0] in {"python", "python3"}:
            cmd = [sys.executable, *cmd[1:]]
        if not is_server_process():
            with _lock:
                bundle = _extensions.setdefault(self._skill, _ExtensionRegistrations())
                _record_companion_name(bundle, f"worker-skip:{clean_name}")
            return
        supervisor = get_global_supervisor()
        if supervisor is None:
            raise ExtensionRegistrationError("companion supervisor is not initialized")
        base_env = _scrub_env(
            list(self._env_allow),
            self._state_dir,
            self._skill,
            granted_keys=list(self._granted_upper),
        )
        reserved_env = {"HOST_SERVICE_TOKEN", "HOST_SERVICE_URL"}
        for key, value in (spec.get("env") or {}).items():
            key_text = str(key)
            if key_text.upper() in FORBIDDEN_EXTENSION_SETTINGS or key_text.upper() in reserved_env:
                continue
            base_env[key_text] = str(value)
        token = self.get_skill_token()
        base_env["HOST_SERVICE_TOKEN"] = token.use_in_request()
        from ouroboros.gateway.host_service import DEFAULT_HOST_SERVICE_HOST, host_service_port
        base_env["HOST_SERVICE_URL"] = f"http://{DEFAULT_HOST_SERVICE_HOST}:{host_service_port()}"
        if self._skill_dir is not None:
            site_dirs = [str(path) for path in _isolated_python_site_dirs(self._skill_dir)]
            if site_dirs:
                existing_pythonpath = base_env.get("PYTHONPATH")
                base_env["PYTHONPATH"] = os.pathsep.join(
                    [*site_dirs, existing_pythonpath] if existing_pythonpath else site_dirs
                )
        workdir = self._runtime_skill_dir or self._skill_dir or self._state_dir
        descriptor = CompanionDescriptor(
            skill_name=self._skill,
            name=clean_name,
            command=cmd,
            cwd=workdir,
            env=base_env,
            ports=[int(port) for port in (spec.get("ports") or []) if str(port).isdigit()],
            restart_policy=str(spec.get("restart_policy") or "on_failure"),
            max_restarts=max(0, int(spec.get("max_restarts") or 5)),
        )
        supervisor.start(descriptor)
        with _lock:
            bundle = _extensions.setdefault(self._skill, _ExtensionRegistrations())
            _record_companion_name(bundle, clean_name)

    def subscribe_event(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> str:
        _reject_extension_child_side_effect("subscribe_event")
        self._require("subscribe_event")
        topic = str(topic or "").strip()
        if topic not in self._subscribe_events:
            raise ExtensionRegistrationError(
                f"skill {self._skill!r} cannot subscribe to undeclared topic {topic!r}"
            )
        sub_id = get_global_event_bus().subscribe(self._skill, topic, self._wrap_runtime_handler(handler))
        with _lock:
            _extensions.setdefault(self._skill, _ExtensionRegistrations()).event_subscriptions.append(sub_id)
        return sub_id

    def send_ws_message(self, message_type: str, data: Dict[str, Any]) -> None:
        _reject_extension_child_side_effect("send_ws_message")
        if "ws_handler" not in self._permissions:
            raise ExtensionRegistrationError(
                f"skill {self._skill!r} cannot 'ws_handler' "
                f"— manifest permissions={sorted(self._permissions)}"
            )
        short = _assert_ws_message_type(message_type)
        with _lock:
            if self._runtime_closing or self._runtime_closed or self._skill in _unloading:
                return
        if current_execution_mode() is ExecutionMode.OUT_OF_PROCESS:
            # Out-of-process: relay through the Host Service loopback bridge (identity
            # re-derived from the token, host-side re-namespacing). The relay touches
            # no shared host state, so it runs OUTSIDE _api_lock — a slow/unreachable
            # host must not block the lock on the loopback HTTP call.
            self._send_ws_message_via_host(short, dict(data or {}))
            return
        full = extension_surface_name(self._skill, short)
        payload = {"type": full, "data": dict(data or {}), "skill": self._skill}
        with self._api_lock:
            broadcaster = _ws_broadcaster
            if broadcaster is None:
                log.debug("extension %s dropped WS message %s: no broadcaster", self._skill, full)
                return
            try:
                broadcaster(payload)
            except Exception:
                log.warning("extension %s WS broadcast failed for %s", self._skill, full, exc_info=True)

    def _send_ws_message_via_host(self, short: str, data: Dict[str, Any]) -> None:
        """Best-effort WS push from an out-of-process child/companion via Host Service."""
        base_url = (os.environ.get("HOST_SERVICE_URL") or "").strip()
        token = (os.environ.get("HOST_SERVICE_TOKEN") or "").strip()
        if not base_url or not token:
            log.debug("extension %s dropped WS message %s: no host bridge env", self._skill, short)
            return
        body = json.dumps({"message_type": short, "data": data}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/ui/ws-message",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "x-skill-token": token},
        )
        try:
            with urllib.request.urlopen(request, timeout=2):  # noqa: S310 - loopback Host Service
                return
        except Exception:
            log.debug("extension %s host WS relay failed for %s", self._skill, short, exc_info=True)

    def on_unload(self, callback: Callable[[], Any]) -> None:
        _reject_extension_child_side_effect("on_unload")
        if not callable(callback):
            raise ExtensionRegistrationError("on_unload callback must be callable")
        with _lock:
            if self._registration_closed or self._runtime_closing or self._runtime_closed or self._skill in _unloading:
                raise ExtensionRegistrationError(
                    f"skill {self._skill!r} cannot register unload callbacks after unload has started"
                )
            # Wrap so an out-of-process isolated-dep extension's cleanup runs with its
            # isolated deps on sys.path at child teardown (true OOP on_unload parity);
            # in-process no-dep extensions get the callback unchanged.
            _extensions.setdefault(self._skill, _ExtensionRegistrations()).unload_callbacks.append(
                self._wrap_runtime_handler(callback)
            )

    def _close_registration(self) -> None:
        with _lock:
            self._registration_closed = True

    def _close_runtime_access(self) -> None:
        with _lock:
            self._registration_closed = True
            self._runtime_closing = True
        with self._api_lock:
            with _lock:
                self._runtime_closed = True

    # --- runtime access ---

    def log(self, level: str, message: str, **fields: Any) -> None:
        lvl = str(level or "info").lower()
        levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
        log.log(
            levels.get(lvl, 20),
            "[ext %s] %s %s",
            self._skill,
            message,
            fields if fields else "",
        )

    def get_settings(self, keys: Sequence[str]) -> Dict[str, Any]:
        with self._api_lock:
            with _lock:
                if self._runtime_closing or self._runtime_closed or self._skill in _unloading:
                    return {}
            if "read_settings" not in self._permissions:
                # Missing permission fails closed without leaking key presence.
                return {}
            settings = self._settings_reader() or {}
            with _lock:
                if self._runtime_closing or self._runtime_closed or self._skill in _unloading:
                    return {}
            out: Dict[str, Any] = {}
            protected_upper = {k.upper() for k in FORBIDDEN_EXTENSION_SETTINGS}
            protected_upper.update(requested_core_setting_keys(list(self._env_allow)))
            for raw_key in keys or ():
                key = str(raw_key).strip()
                canonical = key.upper()
                if not key:
                    continue
                if canonical in protected_upper and canonical not in self._granted_upper:
                    # Do not reveal forbidden/core key presence without a grant.
                    continue
                if key not in self._env_allow and canonical not in self._env_allow_upper:
                    continue
                settings_key = canonical if canonical in protected_upper else key
                if settings_key in settings:
                    out[settings_key] = settings[settings_key]
            return out

    def get_state_dir(self) -> str:
        return str(self._state_dir)

    def skill_job_dir(self, job_id: str) -> pathlib.Path:
        raw = str(job_id or "").strip()
        safe = "".join(
            ch if ch.isalnum() or ch in "-_." else "_"
            for ch in raw
        ).strip("._")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        prefix = (safe or "_job")[:55].rstrip("._-") or "_job"
        safe = f"{prefix}-{digest}"
        root = self._state_dir / "jobs" / safe
        for child in ("assets", "output", "tmp"):
            (root / child).mkdir(parents=True, exist_ok=True)
        return root

    def get_skill_token(self) -> SkillToken:
        return SkillToken(mint_skill_token(self._state_dir, self._skill, self._skill_dir))

    def get_runtime_info(self) -> Dict[str, Any]:
        """Return the PluginAPI runtime-info snapshot without manifest I/O."""
        try:
            from ouroboros.config import (
                get_runtime_mode as _get_runtime_mode,
                DATA_DIR as _DATA_DIR,
            )
            runtime_mode = _get_runtime_mode()
            data_dir = str(_DATA_DIR)
        except Exception:
            runtime_mode = "advanced"
            data_dir = ""
        try:
            from ouroboros import get_version as _get_version
            app_version = str(_get_version())
        except Exception:
            app_version = ""
        try:
            from ouroboros.config import AGENT_SERVER_PORT as _agent_port, PORT_FILE as _PORT_FILE
            server_port = 0
            try:
                port_text = pathlib.Path(_PORT_FILE).read_text(encoding="utf-8").strip()
                if port_text:
                    server_port = int(port_text)
            except Exception:
                server_port = 0
            if server_port <= 0:
                server_port = int(_agent_port)
        except Exception:
            server_port = 0
        skill_dir = str(getattr(self, "_skill_dir", "") or "")
        mode = current_execution_mode()
        return {
            "runtime_mode": runtime_mode,
            "app_version": app_version,
            "data_dir": data_dir,
            "skill_dir": skill_dir,
            "state_dir": str(self._state_dir),
            "server_port": server_port,
            # Capability negotiation: an extension can branch on its execution mode
            # instead of calling an unavailable capability and aborting register().
            "execution_mode": mode.value,
            "capabilities": sorted(available_capabilities(mode)),
        }


# Loader.


def _plugin_entry_path(skill: LoadedSkill) -> Optional[pathlib.Path]:
    """Resolve manifest.entry inside the skill directory."""
    entry = str(skill.manifest.entry or "").strip()
    if not entry:
        return None
    candidate = (skill.skill_dir / entry).resolve()
    try:
        candidate.relative_to(skill.skill_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _module_key(skill_name: str) -> str:
    digest = hashlib.sha1(str(skill_name or "").encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"ouroboros._extensions.m_{digest}"


def _purge_extension_bytecode(skill_dir: pathlib.Path) -> None:
    """Drop bytecode so rapid edits reload fresh source."""
    for pycache in skill_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)


def _stage_extension_import_tree(
    skill: LoadedSkill,
    *,
    state_dir: pathlib.Path,
    entry_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Stage an extension under a fresh import root to avoid stale module reuse."""
    resolved_root = skill.skill_dir.resolve()
    relative_entry = entry_path.relative_to(resolved_root)
    for path in sorted(skill.skill_dir.rglob("*")):
        if is_skill_cache_path(path, resolved_root):
            continue
        if not path.is_symlink():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(resolved_root)
        except Exception as exc:
            raise RuntimeError(
                f"extension {skill.name!r} contains a symlink that resolves outside the skill tree: {path}"
            ) from exc
    child_import_base = os.environ.get("OUROBOROS_EXTENSION_IMPORT_ROOT_BASE", "")
    if os.environ.get("OUROBOROS_EXTENSION_PROCESS_CHILD") == "1" and child_import_base:
        import_root = pathlib.Path(child_import_base) / uuid.uuid4().hex
    else:
        # Tag the staged-tree leaf with the OWNER PID. Under MAX_WORKERS>1 every
        # worker stages concurrently into this SHARED dir; the per-PID prefix lets
        # _sweep_stale_extension_imports tell a peer's still-loading tree (owner
        # alive / fresh) from a real orphan (owner dead + past grace) instead of
        # rmtree-ing a sibling mid-load (which would FileNotFoundError its
        # exec_module and silently drop the skill in that worker).
        import_root = state_dir / "__extension_imports" / f"{os.getpid()}-{uuid.uuid4().hex}"
    staged_skill_dir = import_root / "skill"
    import_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        skill.skill_dir,
        staged_skill_dir,
        ignore=shutil.ignore_patterns(*_SKILL_DIR_CACHE_NAMES),
    )
    _purge_extension_bytecode(staged_skill_dir)
    staged_entry = (staged_skill_dir / relative_entry).resolve()
    staged_entry.relative_to(staged_skill_dir.resolve())
    return import_root, staged_entry


# Grace window before a per-PID staged tree whose owner process is already gone is
# reaped: a just-spawned peer worker can still be mid-copytree of a fresh tree.
# Value-mirrored from supervisor/workers.py:_SPAWN_GRACE_SEC (do NOT import supervisor
# into ouroboros/ — layering inversion); it only affects reclaim latency, not safety.
_IMPORT_SWEEP_GRACE_SEC = 90.0


def _sweep_stale_extension_imports(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    keep: Sequence[pathlib.Path] = (),
) -> None:
    """Remove orphan staged import trees without touching skill state/payload.

    Per-PID safe: staged-tree leaves are named ``<owner_pid>-<uuid>`` (see
    _stage_extension_import_tree), so under MAX_WORKERS>1 — where every worker stages
    into this SHARED dir concurrently — a leaf is reaped ONLY when its owner process is
    dead AND its mtime is past the spawn grace. A peer's still-loading tree (owner
    alive, or fresh within grace) is left alone, so its exec_module never hits a
    FileNotFoundError from a sibling's sweep. Legacy bare-uuid leaves (no parseable
    owner) keep the prior keep-set-only behaviour."""
    root = skill_state_dir(drive_root, skill_name) / "__extension_imports"
    if not root.exists() or not root.is_dir():
        return
    keep_resolved = set()
    for path in keep or ():
        try:
            keep_resolved.add(path.resolve(strict=False))
        except OSError:
            pass
    with _lock:
        bundle = _extensions.get(skill_name)
        if bundle and bundle.import_root:
            try:
                keep_resolved.add(pathlib.Path(bundle.import_root).resolve(strict=False))
            except OSError:
                pass
    try:
        from ouroboros.platform_layer import pid_is_alive as _pid_is_alive
    except Exception:
        _pid_is_alive = None
    now = time.time()
    for child in list(root.iterdir()):
        try:
            resolved = child.resolve(strict=False)
        except OSError:
            resolved = child
        if resolved in keep_resolved:
            continue
        if not child.is_dir():
            continue
        # Cross-process safety (the MAX_WORKERS>1 staging race): only reap a per-PID
        # tree whose OWNER process is DEAD *and* whose mtime is past the spawn grace.
        # Never delete a tree a live (or just-spawned) peer worker is mid-loading.
        owner_pid = None
        try:
            parsed = int(child.name.split("-", 1)[0])
            # A real per-PID leaf is "<pid>-<uuid>" with a plausible PID. A legacy
            # bare-uuid that happens to be all digits would int-parse to a huge number
            # (and OverflowError os.kill); out-of-range -> treat as legacy (fall through
            # to the keep-set reap), never feed an implausible value to pid_is_alive.
            owner_pid = parsed if 0 < parsed < 2_147_483_648 else None
        except (ValueError, IndexError):
            owner_pid = None
        if owner_pid is not None:
            if _pid_is_alive is None:
                continue  # cannot verify liveness -> conservatively keep (never reap unverified)
            try:
                if _pid_is_alive(owner_pid):
                    continue  # owner still running -> tree may be mid-load
                if (now - child.stat().st_mtime) < _IMPORT_SWEEP_GRACE_SEC:
                    continue  # within spawn grace -> a just-spawned peer may be staging
            except Exception:
                continue  # cannot verify liveness/age -> conservative skip (never reap unverified)
        shutil.rmtree(child, ignore_errors=True)


def _extension_runtime_state(
    skill: LoadedSkill,
    *,
    current_hash: str | None = None,
    drive_root: pathlib.Path | None = None,
) -> Dict[str, Any]:
    """Return the liveness authority for one extension."""
    from ouroboros.config import get_runtime_mode

    hash_now = current_hash or skill.content_hash
    skill_dir_now = str(skill.skill_dir.resolve())
    review_stale = skill.review.is_stale_for(hash_now)
    with _lock:
        live_bundle = _extensions.get(skill.name)
        live_loaded = bool(
            live_bundle
            and live_bundle.content_hash == hash_now
            and live_bundle.skill_dir == skill_dir_now
        )
        loaded_present = live_bundle is not None
        load_failure = _load_failures.get(skill.name)
        matched_failure = bool(
            load_failure
            and load_failure.content_hash == hash_now
            and load_failure.skill_dir == skill_dir_now
        )

    review_gate = skill_review_gate(skill.review.status, stale=review_stale)
    if drive_root is None:
        drive_root = pathlib.Path(skill.skill_dir).parent.parent.parent
    grant_status = grant_status_for_skill(pathlib.Path(drive_root), skill)
    grants_usable = bool(grant_status.get("usable", True))
    reason = "ready"
    desired_live = True
    if not skill.manifest.is_extension():
        desired_live = False
        reason = "not_extension"
    elif skill.load_error:
        desired_live = False
        reason = "load_error"
    elif not skill.enabled:
        desired_live = False
        reason = "disabled"
    elif not review_gate["executable_review"]:
        desired_live = False
        reason = review_gate["blocking_reason"]
    elif not grants_usable:
        desired_live = False
        reason = "missing_grants"
    # Light mode allows reviewed skills; it only gates repo mutation/escalation.
    elif matched_failure:
        reason = "load_error"

    return {
        "skill": skill.name,
        "type": skill.manifest.type,
        "runtime_mode": get_runtime_mode(),
        "enabled": skill.enabled,
        "review_status": skill.review.status,
        "review_stale": review_stale,
        "review_gate": review_gate,
        "executable_review": review_gate["executable_review"],
        "grant_status": grant_status,
        "load_error": skill.load_error or (load_failure.error if matched_failure and load_failure else None),
        "desired_live": desired_live,
        "live_loaded": live_loaded,
        "loaded_present": loaded_present,
        "loaded_matches_current": live_loaded,
        "reason": reason,
    }


def _deps_block_reason(drive_root: pathlib.Path, skill: LoadedSkill) -> str:
    """Return the dependency block reason, if live dispatch must refuse load."""
    try:
        from ouroboros.marketplace.install_specs import install_specs_hash
        from ouroboros.marketplace.isolated_deps import read_deps_state
        from ouroboros.skill_dependencies import auto_install_specs_for_skill

        auto_specs = auto_install_specs_for_skill(drive_root, skill)
        if not auto_specs:
            return ""
        deps_state = read_deps_state(drive_root, skill.name, skill.skill_dir)
        status = str(deps_state.get("status") or "")
        if status != "installed":
            if status == "stale":
                return "deps_stale"
            return "deps_failed" if status == "failed" else "deps_missing"
        if deps_state.get("specs_hash") != install_specs_hash(auto_specs):
            return "deps_stale"
        return ""
    except Exception:
        log.debug("extension deps readiness probe failed", exc_info=True)
        return ""


def _apply_deps_block(state: Dict[str, Any], drive_root: pathlib.Path, skill: LoadedSkill) -> Dict[str, Any]:
    if state.get("desired_live"):
        deps_reason = _deps_block_reason(pathlib.Path(drive_root), skill)
        if deps_reason:
            state.update(desired_live=False, reason=deps_reason, load_error=deps_reason)
    return state


def runtime_state_for_skill_name(
    skill_name: str,
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
) -> Dict[str, Any]:
    from ouroboros.config import get_skills_repo_path

    resolved_repo_path = get_skills_repo_path() if repo_path is None else repo_path
    skill = find_skill(drive_root, skill_name, repo_path=resolved_repo_path)
    if skill is None:
        with _lock:
            live_loaded = skill_name in _extensions
        return {
            "skill": skill_name,
            "type": "extension",
            "runtime_mode": "",
            "enabled": False,
            "review_status": "missing",
            "review_stale": True,
            "load_error": "skill not found",
            "desired_live": False,
            "live_loaded": live_loaded,
            "loaded_present": live_loaded,
            "loaded_matches_current": False,
            "reason": "missing",
        }
    return _apply_deps_block(_extension_runtime_state(skill, drive_root=pathlib.Path(drive_root)), pathlib.Path(drive_root), skill)


def runtime_state_for_loaded_skill(skill: "LoadedSkill", drive_root: pathlib.Path | None = None) -> Dict[str, Any]:
    """Runtime state for an already-discovered skill; avoids repeated FS walks."""
    state = _extension_runtime_state(skill, drive_root=pathlib.Path(drive_root) if drive_root is not None else None)
    return _apply_deps_block(state, pathlib.Path(drive_root), skill) if drive_root is not None else state


def is_extension_live(
    skill_name: str,
    drive_root: pathlib.Path,
    *,
    repo_path: str | None = None,
) -> bool:
    state = runtime_state_for_skill_name(skill_name, drive_root, repo_path=repo_path)
    return bool(state.get("desired_live")) and bool(state.get("live_loaded"))


def _revert_enabled_after_load_error(
    revert: bool, drive_root: pathlib.Path, skill_name: str, state: Dict[str, Any]
) -> None:
    """Atomic enable: revert enabled.json to False when an enable-time load fails.

    Shared by every enable path (UI toggle, agent toggle_skill, post-review
    auto-enable) so a skill is never left enabled-but-broken regardless of who
    enabled it.
    """
    if not revert:
        return
    try:
        from ouroboros.skill_loader import save_enabled

        save_enabled(pathlib.Path(drive_root), skill_name, False)
        state["reverted_enabled"] = True
    except Exception:
        log.debug("Failed to revert enabled for %s after load error", skill_name, exc_info=True)


def reconcile_extension(
    skill_name: str,
    drive_root: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    repo_path: str | None = None,
    retry_load_error: bool = False,
    revert_enabled_on_error: bool = False,
) -> Dict[str, Any]:
    """Reconcile one extension's desired and actual live state.

    ``revert_enabled_on_error`` is set by enable paths so that a failed
    out-of-process catalog/register dry-run reverts the persisted enabled flag.
    """
    lifecycle_lock = _lifecycle_lock_for(skill_name)
    with lifecycle_lock:
        state = runtime_state_for_skill_name(skill_name, drive_root, repo_path=repo_path)
        loaded_present = bool(state.get("loaded_present"))
        was_live = bool(state.get("live_loaded"))
        if retry_load_error and state.get("reason") == "load_error" and not was_live:
            with _lock:
                _load_failures.pop(skill_name, None)
            state = runtime_state_for_skill_name(skill_name, drive_root, repo_path=repo_path)
            loaded_present = bool(state.get("loaded_present"))
            was_live = bool(state.get("live_loaded"))
        elif state.get("reason") == "load_error" and not loaded_present:
            state["action"] = "extension_load_error"
            _revert_enabled_after_load_error(revert_enabled_on_error, drive_root, skill_name, state)
            _request_server_reconcile_if_worker(drive_root, skill_name, reason="reconcile_load_error")
            return state
        if state.get("reason") == "missing" or state.get("reason") == "not_extension":
            if loaded_present:
                unload_extension(skill_name)
            state["action"] = "extension_unloaded" if loaded_present else "extension_inactive"
            state["live_loaded"] = False
            state["loaded_present"] = False
            _request_server_reconcile_if_worker(drive_root, skill_name, reason=str(state.get("reason") or "inactive"))
            return state

        if not state.get("desired_live"):
            if loaded_present:
                unload_extension(skill_name)
            state["action"] = "extension_unloaded" if loaded_present else "extension_inactive"
            state["live_loaded"] = False
            state["loaded_present"] = False
            _request_server_reconcile_if_worker(drive_root, skill_name, reason="desired_disabled")
            return state

        if was_live:
            state["action"] = "extension_already_live"
            if is_server_process():
                state["companions"] = ensure_companions_running(
                    skill_name,
                    drive_root,
                    settings_reader,
                    repo_path=repo_path,
                )
            _request_server_reconcile_if_worker(drive_root, skill_name, reason="already_live")
            return state

        from ouroboros.config import get_skills_repo_path

        resolved_repo_path = get_skills_repo_path() if repo_path is None else repo_path
        loaded = find_skill(drive_root, skill_name, repo_path=resolved_repo_path)
        if loaded is None:
            state["reason"] = "missing"
            state["action"] = "extension_inactive"
            _request_server_reconcile_if_worker(drive_root, skill_name, reason="missing")
            return state
        if loaded_present:
            unload_extension(skill_name)
        try:
            err = load_extension(loaded, settings_reader, drive_root=drive_root)
        except Exception as exc:  # an unexpected raise must still revert enable + record
            log.exception("extension %s reconcile load raised", skill_name)
            err = f"skill {skill_name!r} load failure: {type(exc).__name__}: {exc}"
        if err:
            with _lock:
                _load_failures[skill_name] = _ExtensionLoadFailure(
                    content_hash=loaded.content_hash,
                    skill_dir=str(loaded.skill_dir.resolve()),
                    error=err,
                )
            state["reason"] = "load_error"
            state["load_error"] = err
            state["action"] = "extension_load_error"
            _revert_enabled_after_load_error(revert_enabled_on_error, drive_root, skill_name, state)
            _request_server_reconcile_if_worker(drive_root, skill_name, reason="load_error")
            return state
        refreshed = runtime_state_for_skill_name(skill_name, drive_root, repo_path=resolved_repo_path)
        refreshed["action"] = "extension_loaded"
        _request_server_reconcile_if_worker(drive_root, skill_name, reason="loaded")
        return refreshed


def ensure_companions_running(
    skill_name: str,
    drive_root: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    repo_path: str | None = None,
) -> Dict[str, Any]:
    """Ensure the server supervisor matches a live extension's registered companions.

    ``reconcile_extension`` returns early when server-side surfaces are already live;
    this helper deliberately bypasses that ``was_live`` short-circuit for companions
    only. It starts missing companions that the plugin has already registered in the
    server bundle, and stops companions when the persisted desired state is disabled.
    """
    if not is_server_process():
        return {"action": "not_server", "started": [], "missing": []}
    supervisor = get_global_supervisor()
    if supervisor is None:
        return {"action": "no_supervisor", "started": [], "missing": []}

    drive_root = pathlib.Path(drive_root)
    state = runtime_state_for_skill_name(skill_name, drive_root, repo_path=repo_path)
    if not state.get("desired_live"):
        supervisor.stop_skill(skill_name)
        return {"action": "stopped_disabled", "started": [], "missing": []}
    if not state.get("live_loaded"):
        return {"action": "not_live", "started": [], "missing": []}

    with _lock:
        bundle = _extensions.get(skill_name)
        raw_names = list(bundle.companion_names if bundle is not None else [])
    names: List[str] = []
    for raw in raw_names:
        name = str(raw or "").strip()
        if not name or name.startswith("task:"):
            continue
        if name.startswith("worker-skip:"):
            name = name.split(":", 1)[1].strip()
        if name and name not in names:
            names.append(name)
    if not names:
        return {"action": "no_registered_companions", "started": [], "missing": []}

    snapshot_keys = set((supervisor.snapshot() or {}).keys())
    missing = [name for name in names if f"{skill_name}:{name}" not in snapshot_keys]
    if not missing:
        return {"action": "already_running", "started": [], "missing": []}

    from ouroboros.config import get_skills_repo_path

    resolved_repo_path = get_skills_repo_path() if repo_path is None else repo_path
    skill = find_skill(drive_root, skill_name, repo_path=resolved_repo_path)
    if skill is None:
        return {"action": "missing_skill", "started": [], "missing": missing}
    try:
        from ouroboros.skill_dependencies import auto_install_specs_for_skill

        auto_specs = auto_install_specs_for_skill(drive_root, skill)
    except Exception:
        log.debug("extension dependency spec probe failed for %s", skill.name, exc_info=True)
        auto_specs = []
    if auto_specs:
        deps_reason = _deps_block_reason(drive_root, skill)
        if deps_reason:
            return {
                "action": "deps_not_ready",
                "started": [],
                "missing": missing,
                "reason": deps_reason,
            }
    grant_status = grant_status_for_skill(drive_root, skill)
    if not grant_status.get("all_granted", True):
        return {
            "action": "missing_grants",
            "started": [],
            "missing": missing,
            "missing_keys": list(grant_status.get("missing_keys") or []),
            "missing_permissions": list(grant_status.get("missing_permissions") or []),
        }

    state_dir = skill_state_dir(drive_root, skill.name)
    _spawn_out_of_process_companions(
        skill,
        catalog={"companions": missing},
        state_dir=state_dir,
        settings_reader=settings_reader,
        granted_keys=list(grant_status.get("granted_keys") or []),
        dependency_site_dirs_enabled=bool(auto_specs),
    )
    return {"action": "started_missing", "started": missing, "missing": missing}


def load_extension(
    skill: LoadedSkill,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    drive_root: Optional[pathlib.Path] = None,
    _force_in_process: bool = False,
) -> Optional[str]:
    """Load a fresh-reviewed enabled extension, returning a UI-safe error.

    ``drive_root`` must be explicit; defaulting to owner data would pollute
    tests and alternate-drive runtimes.
    """
    if drive_root is None:
        raise TypeError("load_extension requires explicit drive_root")
    if not skill.manifest.is_extension():
        return f"skill {skill.name!r} is not type=extension"
    if skill.load_error:
        return f"skill {skill.name!r} has load_error: {skill.load_error}"
    if not skill.enabled:
        return f"skill {skill.name!r} is disabled"
    try:
        current_hash = compute_content_hash(
            skill.skill_dir,
            manifest_entry=skill.manifest.entry,
            manifest_scripts=skill.manifest.scripts,
        )
    except SkillPayloadUnreadable as exc:
        return (
            f"skill {skill.name!r} payload unreadable at load time: "
            f"{exc}. Fix filesystem state and re-enable."
        )
    runtime_state = _extension_runtime_state(skill, current_hash=current_hash, drive_root=pathlib.Path(drive_root))
    # Light mode permits reviewed extensions; stale review and other gates remain.
    gate = runtime_state.get("review_gate") or skill_review_gate(
        skill.review.status,
        stale=skill.review.content_hash != current_hash,
    )
    if not gate.get("executable_review", False):
        return (
            f"skill {skill.name!r} must carry a fresh executable review "
            f"(status={skill.review.status!r}, "
            f"stale={skill.review.content_hash != current_hash}, "
            f"reason={gate.get('blocking_reason')})"
        )
    if runtime_state["reason"] == "disabled":
        return f"skill {skill.name!r} is disabled"
    entry_path = _plugin_entry_path(skill)
    if entry_path is None:
        return (
            f"skill {skill.name!r} manifest.entry does not resolve to a "
            "file inside the skill directory"
        )

    drive_root = pathlib.Path(drive_root)
    state_dir = skill_state_dir(drive_root, skill.name)
    child_in_process_load = _force_in_process and os.environ.get("OUROBOROS_EXTENSION_PROCESS_CHILD") == "1"
    if not child_in_process_load:
        _sweep_stale_extension_imports(drive_root, skill.name)
    try:
        from ouroboros.skill_dependencies import auto_install_specs_for_skill

        auto_specs = auto_install_specs_for_skill(pathlib.Path(drive_root), skill)
    except Exception:
        log.debug("extension dependency spec probe failed for %s", skill.name, exc_info=True)
        auto_specs = []
    if auto_specs:
        deps_reason = _deps_block_reason(pathlib.Path(drive_root), skill)
        if deps_reason:
            return f"skill {skill.name!r} cannot load until isolated dependencies are ready: {deps_reason}"

    # Core settings and privileged host capabilities require hash-bound grants.
    grant_status = grant_status_for_skill(pathlib.Path(drive_root), skill)
    if not grant_status.get("all_granted", True):
        missing_bits = []
        if grant_status.get("missing_keys"):
            missing_bits.append(f"keys={grant_status.get('missing_keys')}")
        if grant_status.get("missing_permissions"):
            missing_bits.append(f"permissions={grant_status.get('missing_permissions')}")
        return (
            f"skill {skill.name!r} is missing owner grants for "
            f"{', '.join(missing_bits)}. Grant access from the Skills tab."
        )
    granted_core = list(grant_status.get("granted_keys") or [])
    if not _force_in_process:
        try:
            from ouroboros.extension_process_runner import (
                catalog_extension_surfaces,
                extension_requires_process_isolation,
            )

            if extension_requires_process_isolation(skill.skill_dir, bool(auto_specs)):
                catalog = catalog_extension_surfaces(
                    skill,
                    drive_root=pathlib.Path(drive_root),
                    repo_dir=pathlib.Path(__file__).resolve().parents[1],
                    skills_repo_path=skill.skill_dir.parent,
                )
                _register_out_of_process_surfaces(skill, current_hash=current_hash, catalog=catalog)
                _spawn_out_of_process_companions(
                    skill,
                    catalog=catalog,
                    state_dir=state_dir,
                    settings_reader=settings_reader,
                    granted_keys=granted_core,
                    dependency_site_dirs_enabled=bool(auto_specs),
                )
                return None
        except Exception as exc:
            unload_extension(skill.name)
            log.exception("extension %s failed to catalog out-of-process", skill.name)
            return f"skill {skill.name!r} out-of-process catalog failure: {type(exc).__name__}: {exc}"
    staged_import_root: Optional[pathlib.Path] = None
    module_key = _module_key(skill.name)
    try:
        importlib.invalidate_caches()
        staged_import_root, entry_path = _stage_extension_import_tree(
            skill,
            state_dir=state_dir,
            entry_path=entry_path,
        )
        # Package-style spec preserves relative imports from the staged entry dir.
        spec = importlib.util.spec_from_file_location(
            module_key,
            entry_path,
            submodule_search_locations=[str(entry_path.parent)],
        )
        if spec is None or spec.loader is None:
            return f"skill {skill.name!r}: importlib could not build spec"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_key] = module
        with isolated_site_dirs_scope(skill.skill_dir, enabled=bool(auto_specs)):
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if not callable(register):
                # Sibling imports may already be in sys.modules; purge the package.
                unload_extension(skill.name)
                return (
                    f"skill {skill.name!r} plugin.py does not export a "
                    "register(api) callable"
                )
            api = PluginAPIImpl(_PluginAPIConfig(
                skill_name=skill.name,
                permissions=list(skill.manifest.permissions or []),
                env_allowlist=list(skill.manifest.env_from_settings or []),
                state_dir=state_dir,
                settings_reader=settings_reader,
                granted_keys=granted_core,
                subscribe_events=list(getattr(skill.manifest, "subscribe_events", []) or []),
                companion_processes=list(getattr(skill.manifest, "companion_processes", []) or []),
                skill_dir=skill.skill_dir,
                runtime_skill_dir=(staged_import_root / "skill") if staged_import_root is not None else None,
                dependency_site_dirs_enabled=bool(auto_specs),
            ))
            with _lock:
                bundle = _extensions.get(skill.name)
                if bundle is None:
                    bundle = _ExtensionRegistrations()
                    _extensions[skill.name] = bundle
                bundle.content_hash = current_hash
                bundle.skill_dir = str(skill.skill_dir.resolve())
                bundle.import_root = str(staged_import_root) if staged_import_root is not None else None
                bundle.api_instances.append(api)
                _extension_modules[skill.name] = module
                _load_failures.pop(skill.name, None)
            register(api)
            api._close_registration()
            with _lock:
                bundle = _extensions.get(skill.name)
                for tool_name in list(bundle.tools if bundle else []):
                    if tool_name in _tools:
                        _tools[tool_name]["skills_repo_path"] = str(skill.skill_dir.parent)
    except ExtensionRegistrationError as exc:
        # Registration may be partial; always tear it down.
        unload_extension(skill.name)
        return f"skill {skill.name!r} registration error: {exc}"
    except Exception as exc:
        unload_extension(skill.name)
        log.exception("extension %s failed to load", skill.name)
        return f"skill {skill.name!r} load failure: {type(exc).__name__}: {exc}"
    finally:
        if skill.name not in _extensions:
            if staged_import_root is not None:
                shutil.rmtree(staged_import_root, ignore_errors=True)
    return None


def unload_extension(skill_name: str) -> None:
    lifecycle_lock = _lifecycle_lock_for(skill_name)
    with lifecycle_lock:
        _unload_extension_locked(skill_name)


def _unload_extension_locked(skill_name: str) -> None:
    """Remove one extension's surfaces and purge its package from sys.modules."""
    with _lock:
        bundle = _extensions.pop(skill_name, None)
        _extension_modules.pop(skill_name, None)
        import_root = pathlib.Path(bundle.import_root) if bundle and bundle.import_root else None
        callbacks = list(bundle.unload_callbacks) if bundle else []
        api_instances = list(bundle.api_instances) if bundle else []
        event_subscriptions = list(bundle.event_subscriptions) if bundle else []
        companion_names = list(bundle.companion_names) if bundle else []
        supervised_futures = list(bundle.supervised_futures) if bundle else []
        if bundle:
            _unloading.add(skill_name)
        if bundle:
            for key in bundle.tools:
                _tools.pop(key, None)
            for key in bundle.routes:
                _routes.pop(key, None)
            for key in bundle.ws_handlers:
                _ws_handlers.pop(key, None)
            for key in bundle.ui_tabs:
                _ui_tabs.pop(key, None)
            for key in bundle.settings_sections:
                _settings_sections.pop(key, None)
    bus = get_global_event_bus()
    for sub_id in event_subscriptions:
        bus.unsubscribe(sub_id)
    for future in supervised_futures:
        try:
            future.cancel()
        except Exception:
            log.debug("Failed to cancel supervised task for %s", skill_name, exc_info=True)
    supervisor = get_global_supervisor()
    if supervisor is not None:
        for raw_name in companion_names:
            name = str(raw_name or "")
            if name and not name.startswith(("task:", "worker-skip:")):
                supervisor.stop(skill_name, name)
    for api in api_instances:
        close = getattr(api, "_close_runtime_access", None)
        if callable(close):
            close()
    try:
        for callback in callbacks:
            _run_unload_callback(skill_name, callback)
        prefix = _module_key(skill_name)
        # Copy keys before mutating sys.modules.
        for mod_name in list(sys.modules.keys()):
            if mod_name == prefix or mod_name.startswith(prefix + "."):
                sys.modules.pop(mod_name, None)
        if import_root is not None:
            shutil.rmtree(import_root, ignore_errors=True)
    finally:
        with _lock:
            _unloading.discard(skill_name)


def reload_all(
    drive_root: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    repo_path: str | None = None,
) -> Dict[str, Any]:
    """Refresh all extension liveness and return ``skill: error_or_None``."""
    from ouroboros.extension_health import record_extension_health, status_for_runtime_state

    skills = discover_skills(drive_root, repo_path=repo_path)
    skill_names = {s.name for s in skills if s.manifest.is_extension()}
    with _lock:
        loaded_names = set(_extensions.keys())
    results: Dict[str, Any] = {}
    # Version/commit stamp for the durable health vector (live->broken attribution).
    try:
        from ouroboros.config import read_version as _read_version
        from ouroboros.utils import get_git_info as _get_git_info

        hv_version = str(_read_version())
        hv_sha = _get_git_info(pathlib.Path(__file__).resolve().parents[1])[1]
    except Exception:
        hv_version, hv_sha = "", ""
    regressions: List[Dict[str, Any]] = []
    for gone in loaded_names - skill_names:
        try:
            unload_extension(gone)
            _sweep_stale_extension_imports(drive_root, gone)
        except Exception as exc:
            log.exception("Extension reload cleanup failed for %s; continuing", gone)
            results[gone] = f"{type(exc).__name__}: {exc}"
    for skill in skills:
        if not skill.manifest.is_extension():
            continue
        try:
            _sweep_stale_extension_imports(drive_root, skill.name)
            state = reconcile_extension(
                skill.name,
                drive_root,
                settings_reader,
                repo_path=repo_path,
                retry_load_error=True,
            )
            load_error = state.get("load_error")
            if load_error:
                log.error("Extension reload failed for %s: %s", skill.name, load_error)
            results[skill.name] = load_error or (None if state.get("desired_live") else state.get("reason"))
            try:
                health = record_extension_health(
                    drive_root,
                    skill.name,
                    status=status_for_runtime_state(state),
                    version=hv_version,
                    sha=hv_sha,
                    reason=str(state.get("reason") or ""),
                    load_error=str(state.get("load_error") or ""),
                )
                if health.get("newly_regressed"):
                    regressions.append({
                        "skill": skill.name,
                        "last_known_good_sha": (health.get("last_known_good") or {}).get("sha", ""),
                        "sha": hv_sha,
                        "load_error": str(state.get("load_error") or ""),
                    })
            except Exception:
                log.debug("extension health record failed for %s", skill.name, exc_info=True)
        except Exception as exc:
            log.exception("Extension reload failed for %s; continuing", skill.name)
            error = f"{type(exc).__name__}: {exc}"
            try:
                skill_dir = str(skill.skill_dir.resolve())
            except OSError:
                skill_dir = str(skill.skill_dir)
            with _lock:
                _load_failures[skill.name] = _ExtensionLoadFailure(
                    content_hash=skill.content_hash,
                    skill_dir=skill_dir,
                    error=error,
                )
            results[skill.name] = error
            try:
                record_extension_health(
                    drive_root, skill.name, status="broken",
                    version=hv_version, sha=hv_sha, reason="reconcile_exception", load_error=error,
                )
            except Exception:
                log.debug("extension health record failed for %s", skill.name, exc_info=True)
    if regressions:
        for reg in regressions:
            log.error(
                "Extension regression: %s was live at %s, broken now at %s: %s",
                reg["skill"], (reg.get("last_known_good_sha") or "?")[:12],
                (reg.get("sha") or "?")[:12], reg.get("load_error"),
            )
        try:
            from ouroboros.utils import append_jsonl

            append_jsonl(pathlib.Path(drive_root) / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "extension_regression",
                "git_sha": hv_sha,
                "version": hv_version,
                "regressions": regressions,
            })
        except Exception:
            log.debug("Failed to append extension_regression event", exc_info=True)
    return results


def snapshot() -> Dict[str, Any]:
    """Return a read-only snapshot of live extension surfaces."""
    with _lock:
        return {
            "extensions": sorted(_extensions.keys()),
            "tools": sorted(_tools.keys()),
            "routes": sorted(_routes.keys()),
            "ws_handlers": sorted(_ws_handlers.keys()),
            "ui_tabs": [
                dict(copy.deepcopy(value), key=key)
                for key, value in sorted(_ui_tabs.items())
            ],
            "ui_tabs_pending": [],
            # Settings sections follow the same host-surfaced shape as UI tabs.
            "settings_sections": [
                dict(copy.deepcopy(value), key=key)
                for key, value in sorted(_settings_sections.items())
            ],
        }


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Return the registered extension tool, if any."""
    with _lock:
        return dict(_tools.get(name) or {}) or None


def list_ws_handlers() -> Dict[str, Any]:
    with _lock:
        return {k: dict(v) for k, v in _ws_handlers.items()}


def list_routes() -> Dict[str, Any]:
    with _lock:
        return {k: dict(v) for k, v in _routes.items()}


def list_companion_names() -> List[str]:
    """Return host-spawnable companion names across loaded extensions.

    Excludes the ``task:`` (supervised-task) and ``worker-skip:`` markers; used by
    the out-of-process catalog so the host can spawn the declared companions.
    """
    with _lock:
        names: List[str] = []
        for bundle in _extensions.values():
            for raw in bundle.companion_names:
                name = str(raw or "")
                if name and not name.startswith(("task:", "worker-skip:")):
                    names.append(name)
        return names


__all__ = [
    "PluginAPIImpl", "is_extension_live", "load_extension", "reconcile_extension",
    "ensure_companions_running", "unload_extension", "reload_all", "runtime_state_for_skill_name", "snapshot",
    "get_tool", "list_ws_handlers", "list_routes", "list_companion_names",
    "current_execution_mode",
]
