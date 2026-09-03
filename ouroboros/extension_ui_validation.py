"""UI render schema validation for extension widgets/settings."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

from ouroboros.contracts.plugin_api import ExtensionRegistrationError, VALID_EXTENSION_ROUTE_METHODS

_EXTENSION_SHORT_MAX = 24
_UI_RENDER_KINDS = {"", "iframe", "declarative", "module"}
_DECLARATIVE_WIDGET_COMPONENTS = {
    "action", "audio", "chart", "code", "file", "form", "gallery", "image",
    "json", "kv", "key_value", "markdown", "poll", "progress", "status",
    "stream", "subscription", "tabs", "table", "video",
    # Host-owned declarative components; no skill-supplied JS.
    "map", "calendar", "kanban", "group", "metric", "callout",
}
_PASSIVE_FORBIDDEN_TYPES = {"form", "action", "poll", "subscription", "stream"}
_MEDIA_TYPES = {"image", "audio", "video", "file"}
_GROUP_LAYOUTS = {"stack", "grid", "cluster"}
_METRIC_TONES = {"neutral", "info", "success", "warning", "danger"}
_CALLOUT_TONES = {"info", "success", "warning", "danger"}
_TABLE_PRESENTATIONS = {"text", "number", "status", "link"}
_SETTINGS_COMPONENTS = {"form", "action", "markdown", "json"}
_DECLARATIVE_MAX_DEPTH = 8
_DECLARATIVE_MAX_NODES = 256
_GRID_COLUMNS_MAX = 4
WIDGET_FRAME_MIN_HEIGHT = 320
WIDGET_FRAME_MAX_HEIGHT = 8192
# Launch policy of a widget card (``render.start``). This tuple is the SSOT for the
# enum; ``gateway/ui_preferences.py`` imports it for the owner's per-card override.
WIDGET_START_MODES = ("auto", "manual", "retain")
_START_MODE_DEFAULTS = {"module": "manual", "iframe": "manual", "declarative": "auto"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _assert_ws_message_type(message_type: str) -> str:
    candidate = _text(message_type)
    if not candidate:
        raise ExtensionRegistrationError("ws message_type must be non-empty")
    if len(candidate) > _EXTENSION_SHORT_MAX:
        raise ExtensionRegistrationError(f"ws message_type must be <= {_EXTENSION_SHORT_MAX} characters: {candidate!r}")
    if not candidate.replace("_", "").isalnum():
        raise ExtensionRegistrationError(f"ws message_type must be alnum/underscore only: {candidate!r}")
    return candidate


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_optional_text(component: Dict[str, Any], key: str, path: str) -> None:
    if key in component and not isinstance(component.get(key), str):
        raise ExtensionRegistrationError(f"{path}.{key} must be text")


def _validate_component_identity(
    component: Dict[str, Any],
    *,
    path: str,
    state: Dict[str, Any],
) -> None:
    if "id" not in component:
        return
    component_id = component.get("id")
    if not isinstance(component_id, str) or not component_id.strip():
        raise ExtensionRegistrationError(f"{path}.id must be non-empty text")
    component_id = component_id.strip()
    seen_ids = state["ids"]
    prior_path = seen_ids.get(component_id)
    if prior_path is not None:
        raise ExtensionRegistrationError(
            f"{path}.id duplicates component id {component_id!r} first used at {prior_path}"
        )
    seen_ids[component_id] = path


def _validate_grid_columns(value: Any, *, path: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= _GRID_COLUMNS_MAX
    ):
        raise ExtensionRegistrationError(
            f"{path} must be an integer from 1 to {_GRID_COLUMNS_MAX}"
        )


def _validate_start_mode(render: Dict[str, Any], *, kind: str) -> None:
    """Normalize ``render.start``: fill the per-kind default, reject unknown values.

    Only an absent, ``None``, or blank ``start`` takes the default. Any other present
    value must be one of ``WIDGET_START_MODES`` after trimming: the enum is closed, so
    ``0``, ``False``, ``[]``, ``{}`` and ``"Retain"`` are rejected rather than defaulted.
    """
    raw = render.get("start")
    mode = raw.strip() if isinstance(raw, str) else raw
    if mode is None or mode == "":
        default = _START_MODE_DEFAULTS.get(kind)
        if default is not None:
            render["start"] = default
        return
    if mode not in WIDGET_START_MODES:
        raise ExtensionRegistrationError(
            f"ui render start {mode!r} is unsupported; expected one of {list(WIDGET_START_MODES)}"
        )
    if kind == "declarative" and mode != "auto":
        raise ExtensionRegistrationError(
            f"ui render start {mode!r} applies to module and iframe widgets only; "
            "a declarative widget is drawn by the host and has nothing to start"
        )
    render["start"] = mode


def _validate_frame_geometry(render: Dict[str, Any], *, kind: str) -> None:
    """Normalize the bounded geometry shared by framed widget renderers."""
    for key in ("height", "max_height"):
        if key not in render:
            continue
        value = render.get(key)
        if not _is_finite_number(value):
            raise ExtensionRegistrationError(
                f"ui render {key} must be a finite number from "
                f"{WIDGET_FRAME_MIN_HEIGHT} to {WIDGET_FRAME_MAX_HEIGHT}"
            )
        numeric = float(value)
        if not WIDGET_FRAME_MIN_HEIGHT <= numeric <= WIDGET_FRAME_MAX_HEIGHT:
            raise ExtensionRegistrationError(
                f"ui render {key} must be from {WIDGET_FRAME_MIN_HEIGHT} "
                f"to {WIDGET_FRAME_MAX_HEIGHT}"
            )
        normalized = int(round(numeric))
        render[key] = normalized
    if kind == "iframe" and "max_height" in render:
        raise ExtensionRegistrationError("ui render max_height is supported for module widgets only")
    height = render.get("height")
    maximum = render.get("max_height")
    if height is not None and maximum is not None and height > maximum:
        raise ExtensionRegistrationError("ui render height cannot exceed max_height")


def _validate_form_fields(component: Dict[str, Any], *, path: str) -> None:
    fields = component.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ExtensionRegistrationError(f"{path}.fields must be a non-empty list")
    columns = component.get("columns", 1)
    if "columns" in component:
        _validate_grid_columns(columns, path=f"{path}.columns")
    for field_idx, field in enumerate(fields):
        field_path = f"{path}.fields[{field_idx}]"
        if not isinstance(field, dict) or not _text(field.get("name")):
            raise ExtensionRegistrationError(f"{field_path}.name is required")
        for text_key in ("label", "placeholder", "help"):
            _validate_optional_text(field, text_key, field_path)
        if "span" in field:
            _validate_grid_columns(field.get("span"), path=f"{field_path}.span")
            if isinstance(columns, int) and field.get("span") > columns:
                raise ExtensionRegistrationError(
                    f"{field_path}.span cannot exceed {path}.columns"
                )
        for numeric_key in ("min", "max", "step"):
            if numeric_key in field and not _is_finite_number(field.get(numeric_key)):
                raise ExtensionRegistrationError(
                    f"{field_path}.{numeric_key} must be a finite number"
                )
        if "step" in field and float(field["step"]) <= 0:
            raise ExtensionRegistrationError(f"{field_path}.step must be greater than zero")
        if "min" in field and "max" in field and float(field["min"]) > float(field["max"]):
            raise ExtensionRegistrationError(f"{field_path}.min cannot exceed {field_path}.max")


def _validate_components(
    components: Any,
    *,
    path: str,
    depth: int,
    passive: bool,
    state: Dict[str, Any],
) -> None:
    if not isinstance(components, list):
        raise ExtensionRegistrationError(f"{path} must be a list")
    for idx, component in enumerate(components):
        component_path = f"{path}[{idx}]"
        if not isinstance(component, dict):
            raise ExtensionRegistrationError(f"{component_path} must be an object")
        if depth > _DECLARATIVE_MAX_DEPTH:
            raise ExtensionRegistrationError(
                f"{component_path} exceeds maximum component depth {_DECLARATIVE_MAX_DEPTH}"
            )
        state["nodes"] += 1
        if state["nodes"] > _DECLARATIVE_MAX_NODES:
            raise ExtensionRegistrationError(
                f"{component_path} exceeds maximum component count {_DECLARATIVE_MAX_NODES}"
            )
        _validate_component(
            component,
            path=component_path,
            depth=depth,
            passive=passive,
            state=state,
        )


def _validate_component(
    component: Dict[str, Any],
    *,
    path: str,
    depth: int,
    passive: bool,
    state: Dict[str, Any],
) -> None:
    component_type = _text(component.get("type"))
    if component_type not in _DECLARATIVE_WIDGET_COMPONENTS:
        raise ExtensionRegistrationError(f"{path} has unsupported type {component_type!r}")
    _validate_component_identity(component, path=path, state=state)
    for text_key in ("title", "description", "condition_key"):
        _validate_optional_text(component, text_key, path)
    if passive and component_type in _PASSIVE_FORBIDDEN_TYPES:
        raise ExtensionRegistrationError(
            f"{path} cannot use interactive type {component_type!r} inside subscription.render"
        )
    if passive and component_type == "kanban" and component.get("on_move") is not None:
        raise ExtensionRegistrationError(
            f"{path}.on_move is not allowed inside subscription.render"
        )
    if component_type in {"form", "action", "poll", "stream"} and not _text(component.get("route") or component.get("api_route")):
        raise ExtensionRegistrationError(f"{path} requires route or api_route")
    if component_type == "subscription":
        event_name = _text(component.get("event") or component.get("message_type"))
        if not event_name:
            raise ExtensionRegistrationError(f"{path} requires event or message_type")
        _assert_ws_message_type(event_name)
        render_children = component.get("render", [])
        if render_children is None:
            render_children = []
            component["render"] = render_children
        _validate_components(
            render_children,
            path=f"{path}.render",
            depth=depth + 1,
            passive=True,
            state=state,
        )
    if component_type == "tabs":
        tabs = component.get("tabs")
        if not isinstance(tabs, list) or not tabs:
            raise ExtensionRegistrationError(f"{path}.tabs must be a non-empty list")
        for tab_idx, tab in enumerate(tabs):
            tab_path = f"{path}.tabs[{tab_idx}]"
            if not isinstance(tab, dict) or not _text(tab.get("label")):
                raise ExtensionRegistrationError(f"{tab_path}.label is required")
            tab_components = tab.get("components", [])
            _validate_components(
                tab_components,
                path=f"{tab_path}.components",
                depth=depth + 1,
                passive=passive,
                state=state,
            )
    if component_type == "group":
        layout = _text(component.get("layout") or "stack")
        if layout not in _GROUP_LAYOUTS:
            raise ExtensionRegistrationError(
                f"{path}.layout {layout!r} is unsupported; expected one of {sorted(_GROUP_LAYOUTS)}"
            )
        if "columns" in component:
            _validate_grid_columns(component.get("columns"), path=f"{path}.columns")
        _validate_components(
            component.get("components"),
            path=f"{path}.components",
            depth=depth + 1,
            passive=passive,
            state=state,
        )
    method = _text(component.get("method") or "GET").upper()
    if method not in VALID_EXTENSION_ROUTE_METHODS:
        raise ExtensionRegistrationError(f"{path}.method {method!r} is unsupported")
    if component_type == "stream" and method != "GET":
        raise ExtensionRegistrationError(f"{path} stream method must be GET")
    if component_type == "form":
        if "disabled" in component and not isinstance(component.get("disabled"), bool):
            raise ExtensionRegistrationError(f"{path}.disabled must be boolean")
        _validate_optional_text(component, "busy_label", path)
        _validate_form_fields(component, path=path)
    for label, key in (("fields", "path"),) if component_type == "kv" else (("columns", "path"),) if component_type == "table" else ():
        fields = component.get(label)
        if not isinstance(fields, list) or not fields:
            raise ExtensionRegistrationError(f"{path}.{label} must be a non-empty list")
        for field_idx, field in enumerate(fields):
            field_path = f"{path}.{label}[{field_idx}]"
            if not isinstance(field, dict) or not _text(field.get(key)):
                raise ExtensionRegistrationError(f"{field_path}.{key} is required")
            if component_type == "table" and "presentation" in field:
                presentation = _text(field.get("presentation"))
                if presentation not in _TABLE_PRESENTATIONS:
                    raise ExtensionRegistrationError(
                        f"{field_path}.presentation {presentation!r} is unsupported; "
                        f"expected one of {sorted(_TABLE_PRESENTATIONS)}"
                    )
    if component_type == "key_value" and not _text(component.get("items_key") or component.get("path")):
        raise ExtensionRegistrationError(f"{path} requires items_key or path")
    if component_type in _MEDIA_TYPES and not any(_text(component.get(key)) for key in ("route", "api_route", "src", "path")):
        raise ExtensionRegistrationError(f"{path} requires media source")
    if component_type == "gallery":
        if "items" in component and not isinstance(component.get("items"), list):
            raise ExtensionRegistrationError(f"{path}.items must be a list")
        for item_idx, item in enumerate(component.get("items") or []):
            item_path = f"{path}.items[{item_idx}]"
            if not isinstance(item, dict):
                raise ExtensionRegistrationError(
                    f"{path} item {item_idx} must be an object ({item_path})"
                )
            item_type = _text(item.get("type") or "image")
            if item_type not in _MEDIA_TYPES:
                raise ExtensionRegistrationError(f"{item_path}.type {item_type!r} is unsupported")
            if not any(_text(item.get(key)) for key in ("route", "api_route", "src", "path")):
                raise ExtensionRegistrationError(f"{item_path} requires media source")
    if component_type == "map":
        tiles_url = _text(component.get("tiles_url"))
        if tiles_url and not (tiles_url.startswith("https://") or tiles_url.startswith("http://localhost") or tiles_url.startswith("http://127.")):
            raise ExtensionRegistrationError(f"{path}.tiles_url must be https or local")
        markers = component.get("markers")
        if markers is not None and not isinstance(markers, list):
            raise ExtensionRegistrationError(f"{path}.markers must be a list")
        for marker_idx, marker in enumerate(markers or []):
            marker_path = f"{path}.markers[{marker_idx}]"
            if not isinstance(marker, dict):
                raise ExtensionRegistrationError(f"{marker_path} must be an object")
            try:
                float(marker.get("lat"))
                float(marker.get("lon"))
            except (TypeError, ValueError) as exc:
                raise ExtensionRegistrationError(f"{marker_path} requires numeric lat/lon") from exc
    if component_type == "calendar":
        items = component.get("items")
        if items is not None and not isinstance(items, list):
            raise ExtensionRegistrationError(f"{path}.items must be a list")
    if component_type == "kanban":
        columns = component.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ExtensionRegistrationError(f"{path}.columns must be a non-empty list")
        for col_idx, col in enumerate(columns):
            col_path = f"{path}.columns[{col_idx}]"
            if not isinstance(col, dict) or not _text(col.get("id") or col.get("label")):
                raise ExtensionRegistrationError(f"{col_path} requires id or label")
        if "on_move" in component:
            on_move = component.get("on_move")
            if not isinstance(on_move, dict) or not _text(on_move.get("route")):
                raise ExtensionRegistrationError(f"{path}.on_move requires {{route}}")
    if component_type == "metric":
        if not _text(component.get("label")):
            raise ExtensionRegistrationError(f"{path}.label is required")
        if "value" not in component and not _text(component.get("path")):
            raise ExtensionRegistrationError(f"{path} requires value or path")
        if "precision" in component:
            precision = component.get("precision")
            if not isinstance(precision, int) or isinstance(precision, bool) or not 0 <= precision <= 12:
                raise ExtensionRegistrationError(f"{path}.precision must be an integer from 0 to 12")
        tone = _text(component.get("tone") or "neutral")
        if tone not in _METRIC_TONES:
            raise ExtensionRegistrationError(
                f"{path}.tone {tone!r} is unsupported; expected one of {sorted(_METRIC_TONES)}"
            )
        _validate_optional_text(component, "unit", path)
    if component_type == "callout":
        if "text" not in component and not _text(component.get("path")):
            raise ExtensionRegistrationError(f"{path} requires text or path")
        if "text" in component and not isinstance(component.get("text"), str):
            raise ExtensionRegistrationError(f"{path}.text must be text")
        tone = _text(component.get("tone") or "info")
        if tone not in _CALLOUT_TONES:
            raise ExtensionRegistrationError(
                f"{path}.tone {tone!r} is unsupported; expected one of {sorted(_CALLOUT_TONES)}"
            )


def validate_ui_render(render: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the browser-hosted widget declaration surface."""
    if not isinstance(render, dict):
        raise ExtensionRegistrationError("ui render must be an object")
    clean = copy.deepcopy(render)
    kind = _text(clean.get("kind"))
    if kind not in _UI_RENDER_KINDS:
        raise ExtensionRegistrationError(f"ui render kind {kind!r} is unsupported; expected one of {sorted(_UI_RENDER_KINDS - {''})}")
    _validate_start_mode(clean, kind=kind)
    if kind in {"iframe", "module"}:
        _validate_frame_geometry(clean, kind=kind)
    if kind == "module":
        entry = _text(clean.get("entry"))
        if not entry:
            raise ExtensionRegistrationError("module widget render requires entry filename (e.g. 'widget.js')")
        if "/" in entry or ".." in entry or entry.startswith(".") or entry.endswith("/"):
            raise ExtensionRegistrationError(f"module widget entry {entry!r} must be a bare filename inside the skill directory")
        if not entry.endswith((".js", ".mjs")):
            raise ExtensionRegistrationError("module widget entry must be a .js / .mjs file")
        clean["entry"] = entry  # normalized once: the loader capture and the module URL agree
        return clean
    if kind == "declarative":
        if "height" in clean or "max_height" in clean:
            raise ExtensionRegistrationError(
                "ui render height/max_height are supported for framed widgets only"
            )
        try:
            schema_version = int(clean.get("schema_version", 1))
        except (TypeError, ValueError) as exc:
            raise ExtensionRegistrationError("declarative widget schema_version must be 1") from exc
        if schema_version != 1:
            raise ExtensionRegistrationError("declarative widget schema_version must be 1")
        components = clean.get("components")
        if not isinstance(components, list):
            raise ExtensionRegistrationError("declarative widget render requires components[]")
        _validate_components(
            components,
            path="components",
            depth=1,
            passive=False,
            state={"nodes": 0, "ids": {}},
        )
        return clean
    return clean


def validate_settings_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Settings' narrow declarative subset through the widget SSOT."""
    if not isinstance(schema, dict):
        raise ExtensionRegistrationError("settings section schema must be an object")
    components = schema.get("components", [])
    if not isinstance(components, list):
        raise ExtensionRegistrationError("settings section components must be a list")
    for idx, component in enumerate(components):
        path = f"components[{idx}]"
        if not isinstance(component, dict):
            raise ExtensionRegistrationError(f"settings section {path} must be an object")
        component_type = _text(component.get("type"))
        if component_type not in _SETTINGS_COMPONENTS:
            raise ExtensionRegistrationError(
                f"settings section {path}.type {component_type!r} is unsupported; "
                f"expected one of {sorted(_SETTINGS_COMPONENTS)}"
            )
    clean = validate_ui_render({
        "kind": "declarative",
        "schema_version": 1,
        "components": components,
    })
    # Settings sections are drawn in place; the widget launch policy does not apply.
    clean.pop("start", None)
    return clean


__all__ = [
    "WIDGET_FRAME_MAX_HEIGHT",
    "WIDGET_FRAME_MIN_HEIGHT",
    "WIDGET_START_MODES",
    "validate_settings_schema",
    "validate_ui_render",
]
