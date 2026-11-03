"""UI render schema validation for extension widgets/settings.

Framed widgets (``module``, ``iframe``): ``height`` and the module-only ``max_height``
are finite numbers within ``WIDGET_FRAME_MIN_HEIGHT``..``WIDGET_FRAME_MAX_HEIGHT``,
rounded to int, with ``height <= max_height``; a declarative widget accepts neither.
``validate_runtime_ui_render`` also accepts a route-less ``iframe`` and returns it
route-less. ``render.start`` defaults per kind via ``_START_MODE_DEFAULTS``
(module and iframe ``manual``, declarative ``auto``).
"""

from __future__ import annotations

import copy
import math
import pathlib
import re
from typing import Any, Dict

from ouroboros.contracts.plugin_api import VALID_EXTENSION_ROUTE_METHODS, ExtensionRegistrationError
from ouroboros.skill_loader import SkillPayloadUnreadable, _iter_payload_files

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
# A living-message card (``tools/cards.py``) has no skill route, live source or download.
_CARD_FORBIDDEN_TYPES = {"poll", "stream", "subscription", "file"}
_CARD_ROUTE_KEYS = ("route", "api_route", "route_prefix", "on_move")
_CARD_ACTION_ID = re.compile(r"[a-z0-9_]{1,40}")
_DECLARATIVE_MAX_DEPTH = 8
_DECLARATIVE_MAX_NODES = 256
_GRID_COLUMNS_MAX = 4
WIDGET_FRAME_MIN_HEIGHT = 320
WIDGET_FRAME_MAX_HEIGHT = 8192
# Launch policy of a widget card (``render.start``). This tuple is the SSOT for the
# enum; ``gateway/ui_preferences.py`` imports it for the owner's per-card override.
WIDGET_START_MODES = ("auto", "manual", "retain")
_START_MODE_DEFAULTS = {"module": "manual", "iframe": "manual", "declarative": "auto"}
# Appearance is an author declaration for framed module surfaces.  Delivery is
# still opt-in through OuroborosWidget.onTheme so legacy payloads remain live.
WIDGET_APPEARANCE_MODES = ("host", "independent", "fixed")


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


def _validate_appearance(render: Dict[str, Any], *, kind: str) -> None:
    """Validate the optional appearance declaration without changing legacy payloads."""
    if "appearance" not in render:
        return
    raw = render.get("appearance")
    appearance = raw.strip() if isinstance(raw, str) else raw
    if kind != "module":
        raise ExtensionRegistrationError(
            "ui render appearance is supported for module widgets only"
        )
    if appearance not in WIDGET_APPEARANCE_MODES:
        raise ExtensionRegistrationError(
            f"ui render appearance {appearance!r} is unsupported; "
            f"expected one of {list(WIDGET_APPEARANCE_MODES)}"
        )
    render["appearance"] = appearance


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
    if state.get("card") is not None:
        _validate_card_component(component, component_type, path=path, actions=state["card"])
    elif component_type in {"form", "action", "poll", "stream"} and not _text(component.get("route") or component.get("api_route")):
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


def _validate_card_component(
    component: Dict[str, Any], component_type: str, *, path: str, actions: Dict[str, Any],
) -> None:
    """CARD mode: a message has no skill route, so commit controls bind ``submit: {id, once?}``."""
    if component_type in _CARD_FORBIDDEN_TYPES:
        raise ExtensionRegistrationError(
            f"{path} type {component_type!r} is not available in a card (no live source or download)"
        )
    items = component.get("items") if component_type == "gallery" and isinstance(component.get("items"), list) else []
    for where, node in ((path, component), *((f"{path}.items[{i}]", item) for i, item in enumerate(items))):
        for key in _CARD_ROUTE_KEYS:
            if isinstance(node, dict) and key in node:
                raise ExtensionRegistrationError(f"{where}.{key} is not available in a card: a card has no skill route")
        if node is not component and isinstance(node, dict) and _text(node.get("type")) == "file":
            raise ExtensionRegistrationError(f"{where}.type 'file' is not available in a card (no download)")
    if component_type not in {"action", "form"}:
        if "submit" in component:
            raise ExtensionRegistrationError(f"{path}.submit binds only action and form components")
        return
    submit = component.get("submit")
    action_id = submit.get("id") if isinstance(submit, dict) else None
    if not isinstance(action_id, str) or not _CARD_ACTION_ID.fullmatch(action_id):
        raise ExtensionRegistrationError(f"{path}.submit.id is required: [a-z0-9_]{{1,40}}, unique in the card")
    if action_id in actions:
        raise ExtensionRegistrationError(
            f"{path}.submit.id duplicates {action_id!r} first used at {actions[action_id]['path']}"
        )
    if not isinstance(submit.get("once", False), bool):
        raise ExtensionRegistrationError(f"{path}.submit.once must be boolean")
    fields = component.get("fields") if isinstance(component.get("fields"), list) else []
    actions[action_id] = {
        "path": path, "kind": component_type, "once": submit.get("once", False),
        # The label the renderer draws (widgets.js defaults), so a press is named as the owner saw it.
        "label": str(component.get("label") or "Run") if component_type == "action"
        else str(component.get("submit_label") or component.get("title") or component.get("label") or "Submit"),
        **({"fields": [_text(f.get("name")) for f in fields if isinstance(f, dict)]} if component_type == "form" else {}),
    }


def validate_card_components(components: Any) -> Dict[str, Any]:
    """Validate a living-message card's declarative tree and index its commit controls.

    The widget catalog, depth and node bounds apply as for a declarative widget; a card
    additionally has no ``poll``/``stream``/``subscription``/``file`` and no route anywhere,
    and each ``action``/``form`` binds ``submit: {id, once?}`` instead. Returns
    ``{"components": <clean copy>, "actions": {id: {kind, label, once, fields?}}}``; an
    ``ExtensionRegistrationError`` names the exact path of the first problem.
    """
    if not isinstance(components, list) or not components:
        raise ExtensionRegistrationError("components must be a non-empty list")
    clean = copy.deepcopy(components)
    actions: Dict[str, Any] = {}
    _validate_components(clean, path="components", depth=1, passive=False,
                         state={"nodes": 0, "ids": {}, "card": actions})
    return {"components": clean,
            "actions": {key: {k: v for k, v in row.items() if k != "path"} for key, row in actions.items()}}


def validate_ui_render(render: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the browser-hosted widget declaration surface."""
    if not isinstance(render, dict):
        raise ExtensionRegistrationError("ui render must be an object")
    clean = copy.deepcopy(render)
    kind = _text(clean.get("kind"))
    if kind != "module" and ("entry_type" in clean or "assets" in clean):
        raise ExtensionRegistrationError("entry_type/assets apply to module widgets only")
    if kind not in _UI_RENDER_KINDS:
        raise ExtensionRegistrationError(f"ui render kind {kind!r} is unsupported; expected one of {sorted(_UI_RENDER_KINDS - {''})}")
    _validate_start_mode(clean, kind=kind)
    _validate_appearance(clean, kind=kind)
    if kind in {"iframe", "module"}:
        _validate_frame_geometry(clean, kind=kind)
    if kind == "iframe" and not _text(clean.get("route")):
        raise ExtensionRegistrationError("iframe widget render requires route (the extension route the frame loads)")
    if kind == "module":
        from ouroboros.extension_widget_assets import validate_widget_files

        validate_widget_files(clean)
        if clean.get("entry_type") == "module":
            return clean
        entry = _text(clean.get("entry"))
        if not entry:
            raise ExtensionRegistrationError("module widget render requires entry filename (e.g. 'widget.js')")
        if "/" in entry or ".." in entry or entry.startswith(".") or entry.endswith("/"):
            raise ExtensionRegistrationError(f"module widget entry {entry!r} must be a bare filename inside the skill directory")
        if re.fullmatch(r"[A-Za-z0-9._-]+", entry) is None:
            raise ExtensionRegistrationError(
                f"module widget entry {entry!r} must use browser-safe characters [A-Za-z0-9._-] only"
            )
        # The suffix is cosmetic for the ENTRY: the host injects it as a classic
        # inline script, so top-level import/export in it will not run. Files the
        # entry pulls in via dynamic import() may be real ES modules.
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


def validate_runtime_ui_render(render: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI while retaining the public contract's route-less iframe shape."""
    if not isinstance(render, dict):
        return validate_ui_render(render)
    if str(render.get("kind") or "").strip() != "iframe" or str(render.get("route") or "").strip():
        return validate_ui_render(render)
    compatible = dict(render)
    compatible["route"] = "legacy-contract-placeholder"
    clean = validate_ui_render(compatible)
    clean.pop("route", None)
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
    "WIDGET_APPEARANCE_MODES",
    "WIDGET_START_MODES",
    "validate_card_components",
    "validate_runtime_ui_render",
    "read_module_sources",
    "validate_settings_schema",
    "validate_ui_render",
]


def read_module_sources(skill_dir: pathlib.Path | None, *entries: str) -> Dict[str, str]:
    """Capture every reviewed ``.js``/``.mjs`` file under the skill directory for
    the declared ``kind: "module"`` ``entries``, keyed by POSIX path relative to it.

    Read once at load and served from memory by the module endpoint, so the
    bytes a browser receives are the bytes the live (reviewed) bundle loaded
    from; a file edited afterwards is not served until the skill reloads. The
    walk is the review-hash surface (``skill_loader._iter_payload_files``:
    dependency/cache directories and symlinks escaping the root are not
    reviewed, so they are not captured) minus dot-prefixed segments. A missing
    or escaping entry, or any JavaScript file that is not UTF-8 text, fails the
    registration loudly — the tab is not live without its sources.
    """
    if skill_dir is None:
        raise ExtensionRegistrationError(f"module widget entries {entries!r} need a skill directory to read from")
    root = pathlib.Path(skill_dir).resolve()
    for entry in entries:
        if not (root / entry).resolve().is_relative_to(root):
            raise ExtensionRegistrationError(f"module widget entry {entry!r} escapes the skill directory")
    try:
        files = _iter_payload_files(root)
    except SkillPayloadUnreadable as exc:
        raise ExtensionRegistrationError(f"module widget sources are unreadable: {exc}") from None
    sources: Dict[str, str] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix not in (".js", ".mjs") or any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            sources[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ExtensionRegistrationError(f"module widget file {rel!r} is not UTF-8 text: {exc}") from None
        except OSError as exc:
            raise ExtensionRegistrationError(f"module widget file {rel!r} is unreadable: {exc}") from None
    for entry in entries:
        if entry not in sources:
            raise ExtensionRegistrationError(f"module widget entry {entry!r} is missing from the skill directory")
    return sources


# Contribution catalog: the projection publishes icons so clients share this vocabulary.
CONTRIBUTION_PLACEMENTS = ("composer.plus", "composer.chips", "message.action")
CONTRIBUTION_PROVIDERS = ("stt", "tts", "conversation", "place")
CONTRIBUTION_RUNTIMES = ("server", "client")
CONTRIBUTION_DESTINATIONS = ("device", "host")  # or vendor:<nonblank name>
CONTRIBUTION_VERBS = ("draft.insert", "skill.request", "open.tab", "notice")
CONTRIBUTION_RENDER_KINDS = ("declarative", "host")
from ouroboros.icon_geometry import HOST_ICON_NAMES, validate_icon

CONTRIBUTION_ICONS = HOST_ICON_NAMES
CONTRIBUTION_AUDIO_FORMATS = ("audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg", "audio/wav")


def _contribution_text(value: Any, field: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{field} must be {'possibly empty ' if empty else 'nonblank '}text of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{field} must be valid Unicode") from None
    return value


def _contribution_path(value: Any, field: str, *, host: bool = False) -> str:
    path = _contribution_text(value, field, 200)
    if (any(c in path for c in "\\\0?#%:") or any(c.isspace() for c in path)
            or any(part in {"", ".", ".."} or part.startswith(".") for part in path.split("/"))):
        raise ValueError(f"{field} must be a plain relative path without traversal, query or fragment")
    if host and (not path.startswith("ui/") or not path.endswith(".js")):
        raise ValueError("render.entry must name a .js module under ui/")
    return path


def _contribution_object(value: Any, field: str, allowed: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or any(key not in allowed for key in value):
        raise ValueError(f"{field} must be a mapping using only {', '.join(allowed)}")
    return dict(value)


def _contribution_int(value: Any, field: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{field} must be an integer from {lo} to {hi}")
    return value


def _contribution_action(value: Any) -> dict:
    action = _contribution_object(value, "action", ("verb", "text", "route", "with", "then"))
    verb = action.get("verb")
    if verb not in CONTRIBUTION_VERBS:
        raise ValueError(f"action.verb must be one of {CONTRIBUTION_VERBS}")
    allowed = {"verb"}
    if verb in ("draft.insert", "notice"):
        allowed.add("text")
        action["text"] = _contribution_text(action.get("text"), "action.text", 4000 if verb == "draft.insert" else 300)
    elif verb == "skill.request":
        allowed.update(("route", "with", "then"))
        action["route"] = _contribution_path(action.get("route"), "action.route")
        action.setdefault("with", "none")
        action.setdefault("then", "keep_draft")
        if action["with"] not in ("none", "draft", "message") or action["then"] not in ("keep_draft", "clear_draft"):
            raise ValueError("skill.request needs with: none|draft|message and then: keep_draft|clear_draft")
    if set(action) - allowed:
        raise ValueError(f"{verb} only accepts {', '.join(sorted(allowed))}")
    return action


def _contribution_source(value: Any) -> dict:
    source = _contribution_object(value, "source", ("route", "refresh_s"))
    source["route"] = _contribution_path(source.get("route"), "source.route")
    source["refresh_s"] = _contribution_int(source.get("refresh_s"), "source.refresh_s", 15, 3600)
    return source


def _place_contribution(item: dict, manifest: Any) -> dict:
    from ouroboros.places import WAKE_FACTS, is_wake_signal

    if not manifest.is_extension() or "presence" not in manifest.permissions:
        raise ValueError("place requires type: extension and permissions: [presence]")
    destination = _contribution_text(item.get("destination"), "destination", 120)
    if not destination.startswith("vendor:") or not destination[7:].strip():
        raise ValueError("place destination must be vendor:<Name>")
    item.setdefault("scope", "account")
    if item["scope"] not in ("account", "conversation"):
        raise ValueError("place scope must be account or conversation")
    supported = set(WAKE_FACTS)
    if "facts" in item:
        if not isinstance(item["facts"], list):
            raise ValueError("place facts must be a list of core facts or labelled kinds")
        supported = set()
        for fact in item["facts"]:
            if isinstance(fact, dict):
                if set(fact) != {"id", "label"} or not (fact.get("id") in WAKE_FACTS or is_wake_signal(fact.get("id"))):
                    raise ValueError("place fact needs a core id or signal.<slug> and label")
                fact["label"] = _contribution_text(fact["label"], "facts.label", 120)
                name = fact["id"]
            elif isinstance(fact, str) and fact in WAKE_FACTS:
                name = fact
            else:
                raise ValueError("place facts must contain core fact names or labelled kinds")
            if name in supported:
                raise ValueError("place facts must have unique ids")
            supported.add(name)
    wakes = item.get("wakes", [])
    if not isinstance(wakes, list) or any(not isinstance(wake, str) or wake not in supported for wake in wakes):
        raise ValueError("place wakes must be a subset of declared facts (core facts when facts is absent)")
    item["wakes"] = list(dict.fromkeys(wakes))
    if "source" in item:
        item["source"] = _contribution_source(item["source"])
    if "settings" in item:
        value = _contribution_text(item["settings"], "settings", 40)
        if not all(char.isascii() and (char.isalnum() or char in "_-") for char in value):
            raise ValueError("settings must name a settings section id")
    if "read_only" in item and item["read_only"] != "provider":
        raise ValueError("read_only must be provider when the provider can withhold writes")
    return item


def _validate_contribution(entry: dict, manifest: Any, skill_dir: pathlib.Path | None) -> dict:
    common = ("id", "label", "title", "icon", "order", "render")
    fields = (("placement", "action", "source") if "placement" in entry else
              ("provides", "destination", "scope", "facts", "wakes", "source", "settings", "read_only") if entry.get("provides") == "place" else
              ("provides", "runtime", "route", "destination", "languages", "formats", "install_size", "max_bytes", "max_seconds"))
    item = _contribution_object(entry, "contribution", common + fields)
    item["label"] = _contribution_text(item.get("label"), "label", 40)
    item["title"] = _contribution_text(item.get("title", ""), "title", 200, empty=True)
    try:
        default_icon = validate_icon(getattr(manifest, "raw_extra", {}).get("icon", ""))
    except ValueError:
        # Staging reports the manifest icon problem; it must not hide every control/place.
        default_icon = ""
    item["icon"] = validate_icon(item.get("icon", default_icon))
    if type(item.get("order", 0)) is not int:
        raise ValueError("order must be an integer")
    item.setdefault("order", 0)
    render = _contribution_object(item.get("render", {}), "render", ("kind", "entry"))
    render.setdefault("kind", "declarative")
    if render["kind"] not in CONTRIBUTION_RENDER_KINDS:
        raise ValueError(f"render.kind must be one of {CONTRIBUTION_RENDER_KINDS}")
    if render["kind"] == "host":
        render["entry"] = _contribution_path(render.get("entry"), "render.entry", host=True)
        if skill_dir is not None:
            target = pathlib.Path(skill_dir) / render["entry"]
            if not target.is_file() or not target.resolve().is_relative_to(pathlib.Path(skill_dir).resolve()):
                raise ValueError("render.entry must exist inside the reviewed skill directory")
    elif "entry" in render:
        raise ValueError("declarative render has no entry")
    item["render"] = render
    if render["kind"] == "host" or item.get("runtime") == "client":
        if not manifest.is_extension() or "page_module" not in manifest.permissions:
            raise ValueError("page code requires type: extension and permissions: [page_module]")
    if item.get("provides") == "place":
        return _place_contribution(item, manifest)
    if "placement" in item:
        if item["placement"] not in CONTRIBUTION_PLACEMENTS:
            raise ValueError(f"placement must be one of {CONTRIBUTION_PLACEMENTS}")
        item["action"] = _contribution_action(item.get("action"))
        if item["placement"] == "composer.chips":
            item["source"] = _contribution_source(item.get("source"))
        elif "source" in item:
            raise ValueError("source is only supported on composer.chips")
        return item
    if item.get("provides") not in CONTRIBUTION_PROVIDERS:
        raise ValueError(f"provides must be one of {CONTRIBUTION_PROVIDERS}")
    if item.get("runtime") not in CONTRIBUTION_RUNTIMES:
        raise ValueError(f"runtime must be one of {CONTRIBUTION_RUNTIMES}")
    destination = _contribution_text(item.get("destination"), "destination", 120)
    if destination not in CONTRIBUTION_DESTINATIONS and not (destination.startswith("vendor:") and destination[7:].strip()):
        raise ValueError("destination must be device, host or vendor:<Name>")
    if item["runtime"] == "server":
        item["route"] = _contribution_path(item.get("route"), "route")
    elif render["kind"] != "host" or "route" in item:
        raise ValueError("client providers need render.kind: host and no server route")
    for field in ("languages", "formats"):
        if field not in item:
            continue
        values = item[field]
        if not isinstance(values, list) or len(values) > 64:
            raise ValueError(f"{field} must be a list of at most 64 strings")
        item[field] = list(dict.fromkeys(_contribution_text(v, field, 120) for v in values))
        if field == "formats" and any(v not in CONTRIBUTION_AUDIO_FORMATS for v in item[field]):
            raise ValueError(f"formats must be a subset of {CONTRIBUTION_AUDIO_FORMATS}")
    if "install_size" in item:
        item["install_size"] = _contribution_text(item["install_size"], "install_size", 120)
    for field, maximum in (("max_bytes", 524288), ("max_seconds", 600)):
        if field in item:
            if item["provides"] != "stt":
                raise ValueError(f"{field} is only supported by stt providers")
            item[field] = _contribution_int(item[field], field, 1, maximum)
    return item


def validate_contributions(manifest: Any, *, skill_dir: pathlib.Path | None = None) -> tuple[list[dict], list[str]]:
    """Drop only invalid entries; problems are author-facing manifest/preflight warnings."""
    clean, problems = [], []
    for entry in getattr(manifest, "contributes", []):
        try:
            clean.append(_validate_contribution(entry, manifest, skill_dir))
        except ValueError as exc:
            problems.append(f"contributes.{entry.get('id', '?')}: {exc}; fix this entry or remove it")
    return clean, problems
