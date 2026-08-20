"""Host-side validation of surface descriptors returned by a child catalog run.

An isolated-dep extension registers its surfaces in a short-lived child process
and reports them back as plain descriptors. The child is outside the host trust
boundary, so every descriptor is re-validated here — namespace, provider-safe
name, method vocabulary, render schema — before anything is installed.
"""

from __future__ import annotations

from typing import Any, Dict

from ouroboros.contracts.plugin_api import ExtensionRegistrationError, VALID_EXTENSION_ROUTE_METHODS
from ouroboros.extension_surface_names import (
    _EXTENSION_NAME_RE,
    _widget_span_from_render,
    extension_name_prefix,
)
from ouroboros.extension_ui_validation import (
    validate_settings_schema as _validate_settings_schema,
    validate_ui_render as _validate_ui_render,
)


def _out_of_process_handler_proxy(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("extension surface is configured for out-of-process dispatch")


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
    item["render"] = _validate_settings_schema(dict(item.get("render") or {}))
    return item
