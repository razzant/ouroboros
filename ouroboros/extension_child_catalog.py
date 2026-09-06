"""Host-side validation of surface descriptors returned by a child catalog run.

An isolated-dep extension registers its surfaces in a short-lived child process
and reports them back as plain descriptors. The child is outside the host trust
boundary, so every descriptor is re-validated here — namespace, provider-safe
name, method vocabulary, render schema — and staged onto the loader's ABI-9
publication snapshot before anything is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ouroboros.contracts.plugin_api import ExtensionRegistrationError, normalize_extension_route_methods
from ouroboros.extension_registry_state import SURFACE_KINDS, _lock
from ouroboros.extension_surface_names import (
    _EXTENSION_NAME_RE,
    _widget_geometry_from_render,
    _widget_span_from_render,
    extension_name_prefix,
)
from ouroboros.extension_ui_validation import (
    validate_runtime_ui_render as _validate_runtime_ui_render,
    validate_settings_schema as _validate_settings_schema,
)

if TYPE_CHECKING:  # pragma: no cover - annotation-only imports
    from ouroboros.extension_plugin_api import PluginAPIImpl
    from ouroboros.skill_loader import LoadedSkill


def _out_of_process_handler_proxy(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("extension surface is configured for out-of-process dispatch")


def _stage_out_of_process_surfaces(
    api: "PluginAPIImpl",
    skill: "LoadedSkill",
    catalog: Dict[str, Any],
) -> None:
    """Validate child-catalog descriptors and stage them on *api*'s snapshot.

    ABI-9: nothing is installed here — every descriptor is validated and
    staged through the same ``_stage_surface_locked`` seam the in-process
    ``register()`` window uses, so the whole out-of-process registration
    (surfaces AND companions) publishes as ONE validate -> swap -> attach
    transaction; a bad catalog publishes NOTHING rather than a prefix.
    """

    from ouroboros.extension_ui_validation import read_module_sources

    def _proxy(item: Dict[str, Any]) -> Dict[str, Any]:
        item["handler"] = _out_of_process_handler_proxy
        item["skill"] = skill.name
        item["out_of_process"] = True
        item["skills_repo_path"] = str(skill.skill_dir.parent)
        return item

    # Module widget sources are disk reads: capture them before the lock and
    # stage them inside it, exactly like the in-process register_ui_tab path.
    # The child's validator already normalized each declared entry.
    entries = sorted({
        str(item["render"].get("entry") or "")
        for item in (catalog.get("ui_tabs") or [])
        if isinstance(item, dict) and isinstance(item.get("render"), dict)
        and str(item["render"].get("kind") or "") == "module"
    } - {""})
    module_sources = read_module_sources(skill.skill_dir, *entries) if entries else {}

    # Per-kind: the descriptor validator, and the descriptor field carrying the
    # registry key (None = the child's own "key" field, for the UI kinds, which
    # dispatch nothing and so need no handler proxy).
    validators = {
        "tools": (_validate_child_tool_descriptor, "name"),
        "routes": (_validate_child_route_descriptor, "path"),
        "ws_handlers": (_validate_child_ws_descriptor, "type"),
        "ui_tabs": (_validate_child_ui_descriptor, None),
        "settings_sections": (_validate_child_settings_descriptor, None),
    }
    with _lock:
        for kind, live, label in SURFACE_KINDS:
            validate, key_field = validators[kind]
            staged = getattr(api._staged, kind)
            for raw in catalog.get(kind) or []:
                item = validate(skill.name, dict(raw or {}))
                key = (
                    str(item.get(key_field) or "") if key_field
                    else str(item.pop("key", "") or "")
                )
                if not key:
                    continue
                api._stage_surface_locked(
                    live, staged, key, _proxy(item) if key_field else item, label,
                )
        api._staged.module_sources.update(module_sources)


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
    item["methods"] = normalize_extension_route_methods(
        item.get("methods") or ("GET",), subject=f"out-of-process route {path!r}",
    )
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
    render = _validate_runtime_ui_render(dict(item.get("render") or {}))
    item["render"] = render
    span = _widget_span_from_render(render)
    item["span"] = span
    item["grid_span"] = span
    item.update(_widget_geometry_from_render(render))
    return item


def _validate_child_settings_descriptor(skill_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    key = str(item.get("key") or "")
    _validate_child_catalog_namespace(skill_name, "settings section", key)
    if not isinstance(item.get("render", {}), dict):
        raise ExtensionRegistrationError(f"out-of-process settings section {key!r} render must be an object")
    item["render"] = _validate_settings_schema(dict(item.get("render") or {}))
    return item
