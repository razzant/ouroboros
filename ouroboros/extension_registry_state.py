"""Process-wide registries of the surfaces live extensions own.

One loaded extension owns one ``_ExtensionRegistrations`` bundle; the per-surface
maps beside it are keyed by the canonical surface name so unload stays
proportional to a single extension. Everything here is mutated in place under
``_lock``, so every reader — the loader, the PluginAPI, the liveness projection —
shares the same objects.
"""

from __future__ import annotations

import pathlib
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence


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
    drive_root: pathlib.Path | None = None
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


def _lifecycle_lock_for(skill_name: str) -> threading.RLock:
    with _lock:
        lock = _lifecycle_locks.get(skill_name)
        if lock is None:
            lock = threading.RLock()
            _lifecycle_locks[skill_name] = lock
        return lock


def _record_companion_name(bundle: _ExtensionRegistrations, name: str) -> None:
    if name not in bundle.companion_names:
        bundle.companion_names.append(name)
