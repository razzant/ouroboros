"""Owner-local UI preference endpoints."""
from __future__ import annotations

import pathlib
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or
from ouroboros.utils import atomic_write_json, read_json_dict

DEFAULT_UI_PREFERENCES: dict[str, Any] = {
    "widget_order": [],
    "nested_subagents_expanded": False,
    # Resizable side sections (0 = use the CSS default). Clamped to sane ranges so
    # a stored value can never collapse or run away with the layout.
    "sidebar_width": 0,
    "project_panel_width": 0,
    # Per-project "last viewed" ISO timestamps ({project_id: ts}); the sidebar shows
    # an unread dot when a project's last_active_at is newer than this. MERGED (not
    # replaced) on POST so a single-project update never wipes the others.
    "project_last_viewed": {},
    # v6.59.0: per-project sidebar visibility ({project_id: true}) — a lightweight
    # owner "hide" that does NOT resurrect the removed status lifecycle (v6.33.0):
    # the registry stays untouched; hiding is pure presentation. MERGED on POST;
    # a `false` value unhides (and is dropped from the stored map).
    "project_hidden": {},
}
_KNOWN_KEYS = frozenset(DEFAULT_UI_PREFERENCES)
_MAX_WIDGET_ORDER_ITEMS = 200
_MAX_WIDGET_KEY_LENGTH = 200
_SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX = 180, 560
_PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX = 320, 1100
_MAX_PROJECT_LAST_VIEWED = 1000
_MAX_PROJECT_ID_LENGTH = 64


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
    if "nested_subagents_expanded" in raw:
        value = raw.get("nested_subagents_expanded")
        if not isinstance(value, bool):
            raise ValueError("nested_subagents_expanded must be a boolean")
        prefs["nested_subagents_expanded"] = value
    if "sidebar_width" in raw:
        prefs["sidebar_width"] = _normalize_width(raw.get("sidebar_width"), _SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX)
    if "project_panel_width" in raw:
        prefs["project_panel_width"] = _normalize_width(raw.get("project_panel_width"), _PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX)
    if "project_last_viewed" in raw:
        value = raw.get("project_last_viewed")
        if value is None:
            prefs["project_last_viewed"] = {}
        elif not isinstance(value, dict):
            raise ValueError("project_last_viewed must be an object of {project_id: timestamp}")
        else:
            cleaned: dict[str, str] = {}
            for pid, ts in list(value.items())[:_MAX_PROJECT_LAST_VIEWED]:
                key = str(pid or "").strip()[:_MAX_PROJECT_ID_LENGTH]
                if not key:
                    continue
                cleaned[key] = str(ts or "")[:40]
            prefs["project_last_viewed"] = cleaned
    if "project_hidden" in raw:
        value = raw.get("project_hidden")
        if value is None:
            prefs["project_hidden"] = {}
        elif not isinstance(value, dict):
            raise ValueError("project_hidden must be an object of {project_id: bool}")
        else:
            hidden: dict[str, bool] = {}
            for pid, flag in list(value.items())[:_MAX_PROJECT_LAST_VIEWED]:
                key = str(pid or "").strip()[:_MAX_PROJECT_ID_LENGTH]
                if not key:
                    continue
                hidden[key] = bool(flag)
            prefs["project_hidden"] = hidden
    return prefs


async def api_ui_preferences_get(request: Request) -> JSONResponse:
    path = pathlib.Path(request_drive_root(request)) / "state" / "ui_preferences.json"
    try:
        return JSONResponse(_normalize_preferences(read_json_dict(path)))
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
        prefs = _normalize_preferences(read_json_dict(path))
        incoming = _normalize_preferences(body, fill_defaults=False)
        # project_last_viewed MERGES (a per-project update must not wipe the others);
        # all other keys replace.
        if "project_last_viewed" in incoming:
            merged = dict(prefs.get("project_last_viewed") or {})
            merged.update(incoming.pop("project_last_viewed"))
            if len(merged) > _MAX_PROJECT_LAST_VIEWED:
                # Keep the most recent entries by timestamp; bound the map.
                merged = dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_PROJECT_LAST_VIEWED])
            prefs["project_last_viewed"] = merged
        if "project_hidden" in incoming:
            # MERGE like last_viewed; a false flag UNHIDES and is dropped from the map.
            hidden = dict(prefs.get("project_hidden") or {})
            hidden.update(incoming.pop("project_hidden"))
            prefs["project_hidden"] = {pid: True for pid, flag in hidden.items() if flag}
        prefs.update(incoming)
    except ValueError as exc:
        return json_error(str(exc), 400)
    atomic_write_json(path, prefs, trailing_newline=True)
    return JSONResponse({"ok": True, **prefs})


__all__ = [
    "DEFAULT_UI_PREFERENCES",
    "api_ui_preferences_get",
    "api_ui_preferences_post",
]
