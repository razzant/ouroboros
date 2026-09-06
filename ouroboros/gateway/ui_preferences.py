"""Owner-local UI preference endpoints."""
from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.extension_ui_validation import WIDGET_START_MODES
from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or
from ouroboros.utils import atomic_write_json, read_json_dict

DEFAULT_UI_PREFERENCES: dict[str, Any] = {
    "widget_order": [],
    # Owner override of a widget card's launch policy, keyed "<skill>:<tab_id>" with a
    # value from the validator's WIDGET_START_MODES. Bounds only: keys are never
    # checked against live widgets, so a temporarily disabled or removed skill keeps
    # the owner's choice instead of losing it with the next discovery.
    "widget_start_mode": {},
    "nested_subagents_expanded": False,
    # Resizable side sections (0 = use the CSS default). Clamped to sane ranges so
    # a stored value can never collapse or run away with the layout.
    "sidebar_width": 0,
    "project_panel_width": 0,
    # Monotonic, server-clamped read cursors. A Project is unread exactly when its
    # durable visible_revision is greater than this value.
    # ABI 7.0 (ABI-3): the retired ``project_last_viewed`` / ``project_hidden``
    # one-minor no-op inputs are gone — an incoming key answers the ordinary
    # unknown-key 400, a stored legacy key is ignored on read and dropped on
    # the next write (``project_seen_revision`` is the replacement).
    "project_seen_revision": {},
}
_KNOWN_KEYS = frozenset(DEFAULT_UI_PREFERENCES)
_MAX_WIDGET_ORDER_ITEMS = 200
_MAX_WIDGET_START_MODE_ITEMS = 200
_MAX_WIDGET_KEY_LENGTH = 200
_SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX = 180, 560
_PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX = 320, 1100
_MAX_PROJECT_CURSORS = 1000
_MAX_PROJECT_ID_LENGTH = 64


@contextmanager
def _preferences_lock(path: pathlib.Path):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    lock_path = path.with_name(path.name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=4.0)
    if fd is None:
        raise TimeoutError(f"could not lock UI preferences: {lock_path}")
    try:
        yield
    finally:
        release_exclusive_file_lock(lock_path, fd)


def _normalize_width(value: Any, lo: int, hi: int) -> int:
    """0 means 'use the CSS default'; any other value is clamped to [lo, hi]."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("width must be an integer")
    if n <= 0:
        return 0
    return max(lo, min(hi, n))


def _normalize_preferences(
    raw: dict[str, Any] | None,
    *,
    fill_defaults: bool = True,
) -> dict[str, Any]:
    prefs = dict(DEFAULT_UI_PREFERENCES) if fill_defaults else {}
    if not isinstance(raw, dict):
        return prefs
    if "widget_order" in raw:
        value = raw.get("widget_order")
        if value is None:
            prefs["widget_order"] = []
        elif not isinstance(value, list):
            raise ValueError("widget_order must be a list of strings")
        else:
            result: list[str] = []
            seen: set[str] = set()
            for item in value[:_MAX_WIDGET_ORDER_ITEMS]:
                if not isinstance(item, str):
                    raise ValueError("widget_order must be a list of strings")
                key = item.strip()
                if not key or len(key) > _MAX_WIDGET_KEY_LENGTH or key in seen:
                    continue
                seen.add(key)
                result.append(key)
            prefs["widget_order"] = result
    if "widget_start_mode" in raw:
        value = raw.get("widget_start_mode")
        if value is None:
            prefs["widget_start_mode"] = {}
        elif not isinstance(value, dict):
            raise ValueError("widget_start_mode must be an object of {widget_key: mode}")
        else:
            modes: dict[str, str] = {}
            for widget_key, mode in list(value.items())[:_MAX_WIDGET_START_MODE_ITEMS]:
                key = str(widget_key or "").strip()
                if not key or len(key) > _MAX_WIDGET_KEY_LENGTH:
                    continue
                # Trim like the validator trims ``render.start``; non-strings still fail below.
                mode = str(mode).strip()
                if mode not in WIDGET_START_MODES:
                    raise ValueError(
                        f"widget_start_mode values must be one of {list(WIDGET_START_MODES)}"
                    )
                modes[key] = mode
            prefs["widget_start_mode"] = modes
    if "nested_subagents_expanded" in raw:
        value = raw.get("nested_subagents_expanded")
        if not isinstance(value, bool):
            raise ValueError("nested_subagents_expanded must be a boolean")
        prefs["nested_subagents_expanded"] = value
    if "sidebar_width" in raw:
        prefs["sidebar_width"] = _normalize_width(raw.get("sidebar_width"), _SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX)
    if "project_panel_width" in raw:
        prefs["project_panel_width"] = _normalize_width(raw.get("project_panel_width"), _PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX)
    if "project_seen_revision" in raw:
        value = raw.get("project_seen_revision")
        if value is None:
            prefs["project_seen_revision"] = {}
        elif not isinstance(value, dict):
            raise ValueError("project_seen_revision must be an object of {project_id: revision}")
        else:
            cleaned: dict[str, int] = {}
            for pid, revision in list(value.items())[:_MAX_PROJECT_CURSORS]:
                key = str(pid or "").strip()[:_MAX_PROJECT_ID_LENGTH]
                if not key:
                    continue
                try:
                    cleaned[key] = max(0, int(revision or 0))
                except (TypeError, ValueError):
                    raise ValueError("project_seen_revision values must be integers")
            prefs["project_seen_revision"] = cleaned
    return prefs


async def api_ui_preferences_get(request: Request) -> JSONResponse:
    drive_root = request_drive_root(request)
    path = pathlib.Path(drive_root) / "state" / "ui_preferences.json"
    try:
        prefs = _normalize_preferences(read_json_dict(path))
        return JSONResponse(prefs)
    except Exception:
        return JSONResponse(dict(DEFAULT_UI_PREFERENCES))


async def api_ui_preferences_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    unknown = sorted(set(body) - _KNOWN_KEYS)
    if unknown:
        return json_error(f"unknown ui preference key: {unknown[0]}", 400)
    drive_root = request_drive_root(request)
    path = pathlib.Path(drive_root) / "state" / "ui_preferences.json"
    try:
        with _preferences_lock(path):
            prefs = _normalize_preferences(read_json_dict(path))
            incoming = _normalize_preferences(body, fill_defaults=False)
            if "project_seen_revision" in incoming:
                from ouroboros.projects_registry import get_project

                merged = dict(prefs.get("project_seen_revision") or {})
                for project_id, requested in incoming.pop("project_seen_revision").items():
                    project = get_project(drive_root, project_id)
                    if project is None:
                        continue
                    current = max(0, int(project.get("visible_revision") or 0))
                    acknowledged = min(max(0, int(requested or 0)), current)
                    merged[project_id] = max(int(merged.get(project_id) or 0), acknowledged)
                if len(merged) > _MAX_PROJECT_CURSORS:
                    # Bound retained cursors by insertion order; active-only writes
                    # ensure tombstones/unknown ids are not newly admitted here.
                    merged = dict(list(merged.items())[-_MAX_PROJECT_CURSORS:])
                prefs["project_seen_revision"] = merged
            prefs.update(incoming)
            atomic_write_json(path, prefs, trailing_newline=True)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except TimeoutError as exc:
        return json_error(str(exc), 503)
    return JSONResponse({"ok": True, **prefs})


__all__ = [
    "DEFAULT_UI_PREFERENCES",
    "api_ui_preferences_get",
    "api_ui_preferences_post",
]
