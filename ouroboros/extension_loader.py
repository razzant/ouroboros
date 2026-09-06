"""Load reviewed in-process ``type: extension`` skills through PluginAPI.

Extensions run inside Ouroboros, so imports are allowed only after a fresh
executable skill review, manifest permissions, and owner grants pass. All
registered surfaces are provider-safe namespaced and tracked per skill so
disable/reload can tear them down and purge modules cleanly.
"""

from __future__ import annotations

import copy
import functools  # noqa: F401
import importlib
import importlib.util
import inspect  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import logging
import os
import pathlib
import re  # noqa: F401
import secrets  # noqa: F401
import shutil
import sys
import threading
import time  # noqa: F401
import urllib.request  # noqa: F401
import uuid  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from types import ModuleType  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Sequence

from ouroboros.contracts.plugin_api import (
    ExtensionRegistrationError,
    ExecutionMode,
    FORBIDDEN_SKILL_SETTINGS,  # noqa: F401
    VALID_EXTENSION_PERMISSIONS,  # noqa: F401
    VALID_EXTENSION_ROUTE_METHODS,  # noqa: F401
    available_capabilities,  # noqa: F401
    capability_available,  # noqa: F401
)
from ouroboros.event_bus import get_global_event_bus
from ouroboros.extension_companion import CompanionDescriptor, get_global_supervisor, is_server_process  # noqa: F401
from ouroboros.extension_ui_validation import (
    _assert_ws_message_type,  # noqa: F401
    WIDGET_FRAME_MAX_HEIGHT,  # noqa: F401
    WIDGET_FRAME_MIN_HEIGHT,  # noqa: F401
    validate_settings_schema as _validate_settings_schema,  # noqa: F401
    validate_ui_render as _validate_ui_render,  # noqa: F401
)
from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME  # noqa: F401
from ouroboros.provider_models import MODEL_PROVIDER_CREDENTIAL_KEYS  # noqa: F401
from ouroboros.extension_isolated_deps import _isolated_python_site_dirs, async_isolated_site_dirs_scope, isolated_site_dirs_scope, is_skill_cache_path  # noqa: F401
from ouroboros.extension_child_catalog import (
    _out_of_process_handler_proxy,  # noqa: F401
    _stage_out_of_process_surfaces,
    _validate_child_catalog_namespace,  # noqa: F401
    _validate_child_route_descriptor,  # noqa: F401
    _validate_child_settings_descriptor,  # noqa: F401
    _validate_child_tool_descriptor,  # noqa: F401
    _validate_child_ui_descriptor,  # noqa: F401
    _validate_child_ws_descriptor,  # noqa: F401
)
from ouroboros.extension_import_staging import (
    _IMPORT_SWEEP_GRACE_SEC,  # noqa: F401
    _module_key,
    _plugin_entry_path,
    _purge_extension_bytecode,  # noqa: F401
    _stage_extension_import_tree,
    _sweep_stale_extension_imports,
)
from ouroboros.extension_liveness import (
    _apply_deps_block,  # noqa: F401
    _apply_durable_extension_health,  # noqa: F401
    _deps_block_reason,
    _extension_runtime_state,
    _revert_enabled_after_load_error,
    is_extension_live,  # noqa: F401
    runtime_state_for_loaded_skill,  # noqa: F401
    runtime_state_for_skill_name,
)
from ouroboros.extension_plugin_api import (
    ExtensionStaleRecoveryError,
    PluginAPIImpl,
    _reject_extension_child_side_effect,  # noqa: F401
    current_execution_mode,
    mint_skill_token,  # noqa: F401
    set_ws_broadcaster,  # noqa: F401
)
from ouroboros.extension_reconcile_queue import (
    # Both directions of the process-split announcement live in the module that
    # owns the durable carriers: worker -> server reconcile request, server ->
    # workers published generation. The loader only says WHEN state changed.
    announce_extension_state_change as _announce_extension_state_change,
)
from ouroboros.extension_registry_state import (
    # The two live read-projections stay importable as ``extension_loader.<name>``
    # (gateway/widgets.py, gateway/extensions.py, tests/test_gateway_widgets.py);
    # their bodies belong to the module that owns ``_ui_tabs``/``_extensions``.
    live_module_sources,  # noqa: F401 — facade re-export (extraction contract)
    live_widget_projection,  # noqa: F401 — facade re-export (extraction contract)
    _ExtensionLoadFailure,
    _ExtensionRegistrations,  # noqa: F401 — facade re-export (extraction contract)
    _PluginAPIConfig,
    _extension_modules,
    _extensions,
    _lifecycle_lock_for,
    _lifecycle_locks,  # noqa: F401
    _load_failures,
    _lock,
    _record_companion_name,  # noqa: F401
    _routes,
    _settings_sections,
    _tools,
    _ui_tabs,
    _unloading,
    _ws_handlers,
    extension_generation_digest,  # noqa: F401 — ABI-9 dispatch-provenance API
    get_tool_stamped as _registry_get_tool_stamped,
)
from ouroboros.extension_surface_names import (
    _EXTENSION_NAME_PREFIX,  # noqa: F401
    _EXTENSION_NAME_RE,  # noqa: F401
    _EXTENSION_SHORT_MAX,  # noqa: F401
    _EXTENSION_SKILL_TOKEN_MAX,  # noqa: F401
    _assert_namespace_path,  # noqa: F401
    _assert_tool_name,  # noqa: F401
    _extension_skill_token,  # noqa: F401
    _widget_geometry_from_render,  # noqa: F401
    _widget_span_from_render,  # noqa: F401
    extension_name_prefix,  # noqa: F401
    extension_surface_name,  # noqa: F401
    parse_extension_surface_name,  # noqa: F401
)
from ouroboros.skill_loader import _SKILL_DIR_CACHE_NAMES, _sanitize_skill_name, LoadedSkill, SkillPayloadUnreadable, compute_content_hash, discover_skills, find_skill, grant_status_for_skill, requested_core_setting_keys, skill_conflict_status, skill_review_gate, skill_state_dir, skill_state_dir_path  # noqa: F401
from ouroboros.skill_token import SkillToken  # noqa: F401
from ouroboros.tools.skill_exec import _scrub_env  # noqa: F401
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso  # noqa: F401

log = logging.getLogger(__name__)


def _publish_out_of_process_registration(
    skill: LoadedSkill,
    *,
    catalog: Dict[str, Any],
    drive_root: pathlib.Path | None = None,
    state_dir: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    granted_keys: Sequence[str],
    dependency_site_dirs_enabled: bool,
    current_hash: str | None = None,
    expected_generation: str | None = None,
    plugin_api_generation: str = "",
) -> None:
    """Publish an out-of-process extension's catalog as ONE staged snapshot.

    ABI-9 (staged protocol): the initial load (``current_hash`` given) stages
    the child catalog's proxy surfaces AND its declared companion spawns on
    one snapshot and publishes them in a single validate -> SWAP -> attach
    transaction — no partially published extension exists between two
    transactions. The one structurally LATER publication — server-side
    companion recovery via ``ensure_companions_running`` — is GENERATION-
    BOUND (``expected_generation``): it publishes only onto the still-live
    bundle whose generation the caller observed, re-stamping every already-
    published descriptor with the freshly minted digest; a vanished bundle or
    a different generation is a typed ``ExtensionStaleRecoveryError`` refusal
    with zero effects — recovery requires a pre-existing live bundle and can
    never create one, so a completed unload/reload is never resurrected. ANY
    OTHER failure — a refused validation (published nothing) or a post-swap
    attach failure (published, must not stay half-alive) — routes through the
    standard dispose+unload path, generation-bound on the recovery form.

    Cataloged companion names are re-validated against the reviewed manifest
    at the host trust boundary before any process is started; the host is the
    server process, so it owns the supervisor and reuses the in-process
    ``register_companion_process`` descriptor build.
    """
    if (current_hash is None) == (expected_generation is None):
        raise ExtensionRegistrationError(
            "out-of-process publication must be exactly one of: initial load "
            "(current_hash) or generation-bound recovery (expected_generation)"
        )
    names = [str(n).strip() for n in (catalog.get("companions") or []) if str(n).strip()]
    declared = {
        str(item.get("name") or "").strip()
        for item in (skill.manifest.companion_processes or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    from ouroboros.contracts.plugin_api import negotiate_plugin_api

    negotiation = negotiate_plugin_api(skill.manifest)
    if not negotiation.ok:
        raise ExtensionRegistrationError(
            f"PluginAPI negotiation refused: {negotiation.error}"
        )
    api = PluginAPIImpl(_PluginAPIConfig(
        skill_name=skill.name,
        permissions=list(skill.manifest.permissions or []),
        env_allowlist=list(skill.manifest.env_from_settings or []),
        state_dir=state_dir,
        settings_reader=settings_reader,
        drive_root=(drive_root or state_dir.parents[2]),
        granted_keys=list(granted_keys),
        subscribe_events=list(getattr(skill.manifest, "subscribe_events", []) or []),
        companion_processes=list(getattr(skill.manifest, "companion_processes", []) or []),
        skill_dir=skill.skill_dir,
        runtime_skill_dir=skill.skill_dir,
        dependency_site_dirs_enabled=dependency_site_dirs_enabled,
        plugin_api_generation=plugin_api_generation or negotiation.generation,
    ))
    try:
        _stage_out_of_process_surfaces(api, skill, catalog)
        for name in names:
            if name not in declared:
                raise ExtensionRegistrationError(
                    f"out-of-process companion {name!r} escaped manifest.companion_processes"
                )
            api.register_companion_process(name)
        api._publish_registrations(
            content_hash=current_hash,
            skill_dir=str(skill.skill_dir.resolve()) if current_hash is not None else None,
            import_root=None,
            require_live_generation=expected_generation,
        )
    except ExtensionStaleRecoveryError:
        # Typed stale-recovery refusal: nothing was published; disposing by
        # skill name here would unload a newer publication — refuse only.
        api._abort_registration()
        raise
    except Exception:
        # A refused validation published nothing (abort discards the staged
        # snapshot); a post-swap attach failure DID publish. Both route
        # through the standard dispose+unload path — generation-bound on the
        # recovery form, so only the publication this call itself swapped in
        # (or validated against) is ever reaped, never a newer one.
        api._abort_registration()
        unload_extension(
            skill.name,
            expected_generation=(
                None if expected_generation is None
                else (api._published_generation or expected_generation)
            ),
        )
        raise


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


def _finalize_extension_reconcile(
    state: Dict[str, Any],
    drive_root: pathlib.Path | None,
    skill_name: str,
    *,
    reason: str,
    health_stamp: tuple[str, str] | None = None,
) -> None:
    """Close one reconcile: announce the change, receipt it, record its health.

    The announcement itself belongs to ``extension_reconcile_queue`` — one seam,
    two directions, decided by which process is calling, never here. Its return
    is projected onto ``state['server_reconcile']``, which names the WORKER
    handoff only ("requested" / "request_failed" / ""): in the server process
    the same seam publishes a generation instead of asking anyone to reconcile,
    so there is no marker to report. The handoff stays asynchronous and
    fail-soft either way. The durable health vector is then recorded from the
    finished state, so live->broken attribution covers every reconcile exit and
    not only the ``reload_all`` sweep; ``health_stamp`` lets a sweep read the
    version/sha once for the whole batch.
    """
    announced = _announce_extension_state_change(drive_root, skill_name, reason=reason)
    state["process"] = "server" if is_server_process() else "worker"
    state["server_reconcile"] = (
        announced if state["process"] == "worker" and announced in ("requested", "request_failed") else ""
    )
    if drive_root is None:
        return
    from ouroboros.extension_health import record_health_for_runtime_state

    if health_stamp is None:
        record_health_for_runtime_state(drive_root, skill_name, state)
    else:
        record_health_for_runtime_state(drive_root, skill_name, state, stamp=health_stamp)


def reconcile_extension(
    skill_name: str,
    drive_root: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    repo_path: str | None = None,
    skills: Optional[List[LoadedSkill]] = None,
    selected_skill: LoadedSkill | None = None,
    retry_load_error: bool = False,
    revert_enabled_on_error: bool = False,
    health_stamp: tuple[str, str] | None = None,
) -> Dict[str, Any]:
    """Reconcile one extension's desired and actual live state.

    ``revert_enabled_on_error`` is set by enable paths so that a failed
    out-of-process catalog/register dry-run reverts the persisted enabled flag.
    """
    lifecycle_lock = _lifecycle_lock_for(skill_name)
    with lifecycle_lock:
        from ouroboros.config import get_skills_repo_path

        resolved_repo_path = get_skills_repo_path() if repo_path is None else repo_path
        peers = list(skills) if skills is not None else discover_skills(
            drive_root, repo_path=resolved_repo_path
        )
        if selected_skill is not None:
            peers = [item for item in peers if item.name != selected_skill.name]
            peers.append(selected_skill)
        state = runtime_state_for_skill_name(
            skill_name,
            drive_root,
            repo_path=resolved_repo_path,
            skills=peers,
        )
        loaded_present = bool(state.get("loaded_present"))
        was_live = bool(state.get("live_loaded"))
        if retry_load_error and state.get("reason") == "load_error" and not was_live:
            with _lock:
                _load_failures.pop(skill_name, None)
            state = runtime_state_for_skill_name(
                skill_name,
                drive_root,
                repo_path=resolved_repo_path,
                skills=peers,
            )
            loaded_present = bool(state.get("loaded_present"))
            was_live = bool(state.get("live_loaded"))
        elif state.get("reason") == "load_error" and not loaded_present:
            state["action"] = "extension_load_error"
            _revert_enabled_after_load_error(revert_enabled_on_error, drive_root, skill_name, state)
            _finalize_extension_reconcile(state, drive_root, skill_name, reason="reconcile_load_error", health_stamp=health_stamp)
            return state
        if state.get("reason") == "missing" or state.get("reason") == "not_extension":
            if loaded_present:
                unload_extension(skill_name)
            state["action"] = "extension_unloaded" if loaded_present else "extension_inactive"
            state["live_loaded"] = False
            state["loaded_present"] = False
            _finalize_extension_reconcile(state, drive_root, skill_name, reason=str(state.get("reason") or "inactive"), health_stamp=health_stamp)
            return state

        if not state.get("desired_live"):
            if loaded_present:
                unload_extension(skill_name)
            state["action"] = "extension_unloaded" if loaded_present else "extension_inactive"
            state["live_loaded"] = False
            state["loaded_present"] = False
            _finalize_extension_reconcile(state, drive_root, skill_name, reason="desired_disabled", health_stamp=health_stamp)
            return state

        if was_live:
            state["action"] = "extension_already_live"
            if is_server_process():
                state["companions"] = ensure_companions_running(
                    skill_name,
                    drive_root,
                    settings_reader,
                    repo_path=repo_path,
                    selected_skill=selected_skill,
                )
            _finalize_extension_reconcile(state, drive_root, skill_name, reason="already_live", health_stamp=health_stamp)
            return state

        safe_name = _sanitize_skill_name(skill_name)
        loaded = next((item for item in peers if item.name == safe_name), None)
        if loaded is None:
            state["reason"] = "missing"
            state["action"] = "extension_inactive"
            _finalize_extension_reconcile(state, drive_root, skill_name, reason="missing", health_stamp=health_stamp)
            return state
        if loaded_present:
            unload_extension(skill_name)
        try:
            err = load_extension(
                loaded,
                settings_reader,
                drive_root=drive_root,
                skills=peers,
                repo_path=resolved_repo_path,
            )
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
            _finalize_extension_reconcile(state, drive_root, skill_name, reason="load_error", health_stamp=health_stamp)
            return state
        refreshed = runtime_state_for_skill_name(
            skill_name,
            drive_root,
            repo_path=resolved_repo_path,
            skills=peers,
        )
        refreshed["action"] = "extension_loaded"
        _finalize_extension_reconcile(refreshed, drive_root, skill_name, reason="loaded", health_stamp=health_stamp)
        return refreshed


def ensure_companions_running(
    skill_name: str,
    drive_root: pathlib.Path,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    repo_path: str | None = None,
    selected_skill: LoadedSkill | None = None,
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
    state = runtime_state_for_skill_name(
        skill_name, drive_root, repo_path=repo_path,
        skills=[selected_skill] if selected_skill is not None else None,
    )
    if not state.get("desired_live"):
        supervisor.stop_skill(skill_name)
        return {"action": "stopped_disabled", "started": [], "missing": []}
    if not state.get("live_loaded"):
        return {"action": "not_live", "started": [], "missing": []}

    with _lock:
        bundle = _extensions.get(skill_name)
        raw_names = list(bundle.companion_names if bundle is not None else [])
        observed_generation = str(bundle.generation_digest or "") if bundle is not None else ""
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
    skill = selected_skill or find_skill(
        drive_root, skill_name, repo_path=resolved_repo_path,
    )
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

    # Fix-round-6: resolved WITHOUT mkdir — the post-fence attach creates it.
    state_dir = skill_state_dir_path(drive_root, skill.name)
    # ABI-9 generation-bound recovery: publish under the lifecycle lock; the
    # seam re-validates that the publication observed above is still live —
    # an unload/reload completing since the snapshot is a typed refusal with
    # zero effects (no bundle resurrection, no filesystem writes).
    with _lifecycle_lock_for(skill_name):
        try:
            _publish_out_of_process_registration(
                skill,
                catalog={"companions": missing},
                drive_root=drive_root,
                state_dir=state_dir,
                settings_reader=settings_reader,
                granted_keys=list(grant_status.get("granted_keys") or []),
                dependency_site_dirs_enabled=bool(auto_specs),
                expected_generation=observed_generation,
            )
        except ExtensionStaleRecoveryError as exc:
            return {
                "action": "stale_recovery_refused",
                "started": [],
                "missing": missing,
                "reason": str(exc),
            }
    return {"action": "started_missing", "started": missing, "missing": missing}


def load_extension(
    skill: LoadedSkill,
    settings_reader: Callable[[], Dict[str, Any]],
    *,
    drive_root: Optional[pathlib.Path] = None,
    skills: Optional[List[LoadedSkill]] = None,
    repo_path: str | None = None,
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
    runtime_state = _extension_runtime_state(
        skill,
        current_hash=current_hash,
        drive_root=pathlib.Path(drive_root),
        skills=skills,
        repo_path=repo_path,
    )
    if runtime_state["reason"] == "skill_conflict":
        conflict_names = list((runtime_state.get("conflict") or {}).get("skills") or [])
        return (
            f"skill {skill.name!r} conflicts with enabled skills "
            f"{conflict_names}. Disable the conflicting skill first."
        )
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
    # ABI-1: negotiate the manifest against this host's PluginAPI BEFORE any
    # plugin import or out-of-process catalog. Absent field -> the LEGACY
    # generation (grandfathered on its existing hash-bound PASS); a declared
    # field is held to the full contract with typed, educational refusals.
    from ouroboros.contracts.plugin_api import negotiate_plugin_api

    negotiation = negotiate_plugin_api(skill.manifest, mode=current_execution_mode())
    if not negotiation.ok:
        return f"skill {skill.name!r} PluginAPI negotiation refused: {negotiation.error}"
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
                # ABI-9: surfaces AND companions publish as ONE staged
                # snapshot; the seam routes any failure to dispose+unload.
                _publish_out_of_process_registration(
                    skill,
                    catalog=catalog,
                    drive_root=pathlib.Path(drive_root),
                    state_dir=state_dir,
                    settings_reader=settings_reader,
                    granted_keys=granted_core,
                    dependency_site_dirs_enabled=bool(auto_specs),
                    current_hash=current_hash,
                    plugin_api_generation=negotiation.generation,
                )
                return None
        except Exception as exc:
            unload_extension(skill.name)
            log.exception("extension %s failed to catalog out-of-process", skill.name)
            return f"skill {skill.name!r} out-of-process catalog failure: {type(exc).__name__}: {exc}"
    staged_import_root: Optional[pathlib.Path] = None
    module_key = _module_key(skill.name)
    api: Optional[PluginAPIImpl] = None
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
                drive_root=pathlib.Path(drive_root),
                granted_keys=granted_core,
                subscribe_events=list(getattr(skill.manifest, "subscribe_events", []) or []),
                companion_processes=list(getattr(skill.manifest, "companion_processes", []) or []),
                skill_dir=skill.skill_dir,
                runtime_skill_dir=(staged_import_root / "skill") if staged_import_root is not None else None,
                dependency_site_dirs_enabled=bool(auto_specs),
                plugin_api_generation=negotiation.generation,
            ))
            if current_execution_mode() is ExecutionMode.IN_PROCESS:
                api._disclose_model_capable_dispatch("register", "register")
            register(api)
            # ABI-9: nothing register() staged is visible yet; publish the
            # whole snapshot atomically (validate -> swap -> attach; deferred
            # side effects attach only after the swap).
            api._publish_registrations(
                content_hash=current_hash,
                skill_dir=str(skill.skill_dir.resolve()),
                import_root=str(staged_import_root) if staged_import_root is not None else None,
            )
            with _lock:
                _extension_modules[skill.name] = module
                bundle = _extensions.get(skill.name)
                for tool_name in list(bundle.tools if bundle else []):
                    if tool_name in _tools:
                        _tools[tool_name]["skills_repo_path"] = str(skill.skill_dir.parent)
    except ExtensionRegistrationError as exc:
        # A refused validation published nothing (abort discards the staged
        # snapshot); a post-swap attach failure DID publish — either way
        # unload_extension is the standard dispose path and reaps whatever
        # the bundle recorded, then the imported package is purged.
        if api is not None:
            api._abort_registration()
        unload_extension(skill.name)
        return f"skill {skill.name!r} registration error: {exc}"
    except Exception as exc:
        if api is not None:
            api._abort_registration()
        unload_extension(skill.name)
        log.exception("extension %s failed to load", skill.name)
        return f"skill {skill.name!r} load failure: {type(exc).__name__}: {exc}"
    finally:
        if skill.name not in _extensions:
            if staged_import_root is not None:
                shutil.rmtree(staged_import_root, ignore_errors=True)
    return None


def unload_extension(skill_name: str, *, expected_generation: str | None = None) -> bool:
    """Unload one extension; a non-None ``expected_generation`` binds disposal
    (ABI-9 recovery failure path) to ONLY the publication whose generation the
    caller observed or made: a missing bundle or a different (newer)
    generation is disclosed and left untouched. Returns whether it ran."""
    lifecycle_lock = _lifecycle_lock_for(skill_name)
    with lifecycle_lock:
        if expected_generation is not None:
            with _lock:
                bundle = _extensions.get(skill_name)
                live_generation = str(bundle.generation_digest or "") if bundle is not None else ""
            if bundle is None or live_generation != str(expected_generation):
                log.warning(
                    "extension %s generation-bound disposal skipped: expected generation %r, live %r",
                    skill_name, str(expected_generation), live_generation or None,
                )
                return False
        _unload_extension_locked(skill_name)
        return True


def _unload_extension_locked(skill_name: str) -> None:
    """Remove one extension's surfaces and purge its package from sys.modules.

    ABI-9 unload visibility (fix-round-3): the extension loses its INPUTS
    first — the event-bus unsubscribe and the runtime-API close happen BEFORE
    the bundle and its surfaces leave the registries, so a publish STARTED
    after the unsubscribe can never deliver into an extension whose surfaces
    are already gone. Residual by design (EventBus copy semantics, see
    ``EventBus.publish``): a publisher that copied the handler under the bus
    lock BEFORE the unsubscribe may still invoke it afterwards; the
    ``_unloading`` latch — set in the same registry-lock hold that snapshots
    the subscription ids, so no new publication can interleave — and the
    closed runtime API make such a late call a no-op against the host.
    """
    with _lock:
        bundle = _extensions.get(skill_name)
        event_subscriptions = list(bundle.event_subscriptions) if bundle else []
        api_instances = list(bundle.api_instances) if bundle else []
        if bundle:
            _unloading.add(skill_name)
    try:
        # 1) Close visibility: after this, a NEW publish finds no subscription
        #    and every runtime API call answers as closed.
        bus = get_global_event_bus()
        for sub_id in event_subscriptions:
            bus.unsubscribe(sub_id)
        for api in api_instances:
            close = getattr(api, "_close_runtime_access", None)
            if callable(close):
                close()
        # 2) Only now remove the bundle and its published surfaces.
        with _lock:
            bundle = _extensions.pop(skill_name, None)
            _extension_modules.pop(skill_name, None)
            import_root = pathlib.Path(bundle.import_root) if bundle and bundle.import_root else None
            callbacks = list(bundle.unload_callbacks) if bundle else []
            companion_names = list(bundle.companion_names) if bundle else []
            supervised_futures = list(bundle.supervised_futures) if bundle else []
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
    from ouroboros.extension_health import fresh_code_stamp, record_health_for_runtime_state

    skills = discover_skills(drive_root, repo_path=repo_path)
    health_stamp = fresh_code_stamp()
    skill_names = {s.name for s in skills if s.manifest.is_extension()}
    with _lock:
        loaded_names = set(_extensions.keys())
    results: Dict[str, Any] = {}
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
                skills=skills,
                retry_load_error=True,
                health_stamp=health_stamp,
            )
            load_error = state.get("load_error")
            if load_error:
                log.error("Extension reload failed for %s: %s", skill.name, load_error)
            results[skill.name] = load_error or (None if state.get("desired_live") else state.get("reason"))
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
                record_health_for_runtime_state(drive_root, skill.name, {
                    "desired_live": True, "reason": "load_error", "load_error": error,
                    "process": "server" if is_server_process() else "worker",
                }, stamp=health_stamp)
            except Exception:
                log.debug("extension health record failed for %s", skill.name, exc_info=True)
    # Whole-set announcement: the per-skill reconciles above miss the pool the
    # "gone" sweep unloaded and an install with no extension skills left at all.
    _announce_extension_state_change(drive_root, "", reason="reload_all")
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
            # Settings sections follow the same host-surfaced shape as UI tabs.
            "settings_sections": [
                dict(copy.deepcopy(value), key=key)
                for key, value in sorted(_settings_sections.items())
            ],
        }


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """The registered tool, if any (ABI-9: legacy descriptors digest-stamped)."""
    return _registry_get_tool_stamped(name)


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
    "live_widget_projection", "live_module_sources",
    "get_tool", "list_ws_handlers", "list_routes", "list_companion_names",
    "current_execution_mode",
]
