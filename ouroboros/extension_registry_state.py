"""Process-wide registries of the surfaces live extensions own.

One loaded extension owns one ``_ExtensionRegistrations`` bundle; the per-surface
maps beside it are keyed by the canonical surface name so unload stays
proportional to a single extension. Everything here is mutated in place under
``_lock``, so every reader — the loader, the PluginAPI, the liveness projection —
shares the same objects.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
    # ``POSIX relative path -> text`` of every reviewed ``.js``/``.mjs`` file under
    # the skill directory, read once at load when a ``kind: "module"`` widget
    # registered; the module endpoint serves these bytes, never the mutable
    # skill directory. Not part of snapshot().
    module_sources: Dict[str, str] = field(default_factory=dict)
    content_hash: Optional[str] = None
    skill_dir: Optional[str] = None
    import_root: Optional[str] = None
    # ABI-9: minted at each atomic publication; stamped into the dispatch
    # surfaces (tools/routes/ws) so physical-call provenance can name the
    # exact published generation it dispatched against.
    generation_digest: str = ""
    # ABI-1: the PluginAPI generation this bundle was negotiated under
    # (manifest ``plugin_api`` field, or the legacy generation for
    # grandfathered field-less payloads).
    plugin_api_generation: str = ""


@dataclass
class _StagedSupervisedTask:
    """A supervised-task request captured during the registration window.

    The asyncio runner is deliberately NOT created here: it starts only at
    publication, after the whole registration validated (ABI-9) — a refused
    registration therefore can never leak a running task outside a bundle.
    """

    name: str
    factory: Callable[[], Any]
    restart_policy: str = "on_failure"
    max_restarts: int = 5
    backoff_seconds: float = 2.0


@dataclass
class _StagedCompanionSpawn:
    """A validated companion descriptor whose spawn is deferred to publication.

    ``spec`` carries the manifest companion entry so the post-fence attach can
    materialize the settings-derived env (fix-round-6): the descriptor's env is
    EMPTY until ``_publish_registrations`` fills it after the generation fence.
    """

    name: str
    descriptor: Any
    spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _StagedEventSubscription:
    """A validated event subscription whose bus attach is deferred to publication.

    ABI-9: ``subscribe_event`` validates the topic and mints the sub_id during
    the registration window but the bus subscription is created only at
    publication — an event published before the snapshot swap can never invoke
    a staged handler (pre-publication invisibility, not eventual cleanup)."""

    sub_id: str
    topic: str
    handler: Callable[[Dict[str, Any]], Any]


@dataclass
class _StagedRegistrations:
    """Private staging area for one PluginAPI registration window (ABI-9).

    ``register()`` accumulates every surface and side-effect request here;
    nothing reaches the process-wide registries until the loader publishes the
    whole snapshot atomically (validate -> swap -> attach). Every deferred
    side effect — supervised runners, companion spawns, event-bus
    subscriptions — attaches only at publication, after the definitive
    unload/conflict validation AND after the snapshot swap (so a handler is
    visible to the bus only for an already-published extension); an aborted
    registration leaves zero residue — staging is purely computational, so
    there is nothing to dispose — and a post-swap attach failure is disposed
    through the standard unload path.
    """

    tools: Dict[str, Any] = field(default_factory=dict)
    routes: Dict[str, Any] = field(default_factory=dict)
    ws_handlers: Dict[str, Any] = field(default_factory=dict)
    ui_tabs: Dict[str, Any] = field(default_factory=dict)
    settings_sections: Dict[str, Any] = field(default_factory=dict)
    unload_callbacks: List[Callable[[], Any]] = field(default_factory=list)
    event_subscriptions: List["_StagedEventSubscription"] = field(default_factory=list)
    companion_names: List[str] = field(default_factory=list)
    supervised_tasks: List[_StagedSupervisedTask] = field(default_factory=list)
    companion_spawns: List[_StagedCompanionSpawn] = field(default_factory=list)
    # Module-widget JavaScript captured during the window (a disk read taken
    # OUTSIDE the lock, like every other staged value: purely computational).
    # It reaches ``_ExtensionRegistrations.module_sources`` only at the swap, so
    # a refused publication never leaves a half-populated bundle serving bytes.
    module_sources: Dict[str, str] = field(default_factory=dict)


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
    # ABI-1: negotiated PluginAPI generation served to this extension
    # ("" -> the loader's negotiation default for grandfathered payloads).
    plugin_api_generation: str = ""


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

# The five surface kinds, each naming the field that carries it on a staged
# snapshot AND on a published bundle, its live registry, and the word used in
# refusals. Every walk over "all surfaces" — conflict detection, the atomic
# swap, the child-catalog staging — reads this, so a sixth kind cannot be
# added to some walks and forgotten by others.
SURFACE_KINDS: Sequence[tuple[str, Dict[str, Any], str]] = (
    ("tools", _tools, "tool"),
    ("routes", _routes, "route"),
    ("ws_handlers", _ws_handlers, "ws handler"),
    ("ui_tabs", _ui_tabs, "ui tab"),
    ("settings_sections", _settings_sections, "settings section"),
)

# Dispatch surfaces: a physical call arrives on these, so each descriptor is
# stamped with the publication that owns it. UI kinds are read as a snapshot
# and carry no per-surface provenance.
DISPATCH_SURFACE_KINDS = frozenset({"tools", "routes", "ws_handlers"})


def _lifecycle_lock_for(skill_name: str) -> threading.RLock:
    with _lock:
        lock = _lifecycle_locks.get(skill_name)
        if lock is None:
            lock = threading.RLock()
            _lifecycle_locks[skill_name] = lock
        return lock


def extension_generation_digest(skill_name: str) -> str:
    """Return the generation digest of the skill's live publication, or ``""``.

    Dispatch-side provenance reads this (or the per-surface stamp) to name the
    exact published registration generation a physical call ran against.
    """
    with _lock:
        bundle = _extensions.get(str(skill_name or ""))
        return str(bundle.generation_digest or "") if bundle is not None else ""


def live_extension_fingerprint() -> str:
    """Digest of WHAT this process has live, comparable across processes.

    ``generation_digest`` is minted fresh per publication, so it can never be
    compared between processes: the server and a task worker importing the very
    same payload mint different digests. What IS identical on both sides is the
    payload identity each loaded — exactly the triple ``live_loaded`` compares
    (skill name, content hash, resolved skill dir) — so the fingerprint is taken
    over that. An EMPTY registry has a definite digest of its own, which is what
    lets a reader tell "the publisher has nothing live" apart from "nothing was
    ever published" (the absent-marker case, which reads as ``""``).
    """
    with _lock:
        rows = sorted(
            (name, str(bundle.content_hash or ""), str(bundle.skill_dir or ""))
            for name, bundle in _extensions.items()
        )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def live_widget_projection(skill_name: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Live UI tabs joined with their owner's revision, under ONE lock.

    Each row is ``{"tab": <snapshot-shaped tab>, "revision": <owner content_hash>}``.
    A single ``_lock`` acquisition, so a reload racing the read can never pair a
    tab with another generation's revision. With ``skill_name`` the rows are that
    skill's only and ``None`` means it has no live bundle; an empty list is a live
    bundle declaring no tabs. Loader-side truth for GET /api/widgets, which must
    not re-discover skills; module sources are read through
    ``live_module_sources``. No skill directory here.
    """
    with _lock:
        if skill_name is not None and skill_name not in _extensions:
            return None
        rows: List[Dict[str, Any]] = []
        for key, value in sorted(_ui_tabs.items()):
            owner = str(value.get("skill") or "")
            if skill_name is not None and owner != skill_name:
                continue
            bundle = _extensions.get(owner)
            rows.append({
                "tab": dict(copy.deepcopy(value), key=key),
                "revision": str(bundle.content_hash or "") if bundle is not None else "",
            })
        return rows


def live_module_sources(skill_name: str) -> Optional[Dict[str, str]]:
    """The reviewed ``.js``/``.mjs`` texts a live bundle captured at load, keyed by
    POSIX path relative to the skill directory; ``None`` when the skill has no
    live bundle (the module endpoint's 409), empty until a module tab registered.
    One ``_lock`` read; the returned dict is a fresh key snapshot, not a byte copy.
    """
    with _lock:
        bundle = _extensions.get(skill_name)
        return None if bundle is None else dict(bundle.module_sources)


def get_tool_with_generation(name: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Detached tool descriptor plus the generation digest it dispatches
    against, read under ONE ``_lock`` hold (ABI-9).

    A legacy descriptor that predates the per-surface ``extension_generation``
    stamp gets the owner bundle's live digest from the SAME registry snapshot
    — a republish between two separate lock acquisitions could otherwise pair
    an old unstamped handler with the NEW generation's digest. Returns
    ``(None, "")`` for an unknown surface."""
    with _lock:
        raw = _tools.get(str(name or ""))
        if not raw:
            return None, ""
        tool = dict(raw)
        digest = str(tool.get("extension_generation") or "")
        if not digest:
            bundle = _extensions.get(str(tool.get("skill") or ""))
            digest = str(bundle.generation_digest or "") if bundle is not None else ""
        return tool, digest


def get_tool_stamped(name: str) -> Optional[Dict[str, Any]]:
    """``get_tool`` body: the detached descriptor with a legacy (unstamped)
    entry pre-stamped from the SAME lock hold's digest — every consumer of the
    loader's ``get_tool`` reads dispatch provenance atomically."""
    tool, digest = get_tool_with_generation(name)
    if tool is not None and digest and not tool.get("extension_generation"):
        tool["extension_generation"] = digest
    return tool


def _record_companion_name(bundle: _ExtensionRegistrations, name: str) -> None:
    if name not in bundle.companion_names:
        bundle.companion_names.append(name)
