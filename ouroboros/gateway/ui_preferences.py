"""Owner-local UI preference endpoints, stored in ``state/ui_preferences.json``.

Keys: ``voice`` (``stt``, ``tts``, ``tts_voice``), ``contributions``, ``widget_order``, ``widget_size``, ``widget_start_mode`` (values from
``extension_ui_validation.WIDGET_START_MODES``), ``nested_subagents_expanded``, the
empty-Main ``welcome`` copy, the two side-section widths, the read cursors and
``read_ahead_revisions``. A POST may carry any subset of keys; an unknown key answers 400.
``shared_read`` (row addresses) is an input only: the host records it as read-ahead.
"""
from __future__ import annotations

import pathlib
import copy
from contextlib import contextmanager
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.extension_ui_validation import WIDGET_START_MODES
from ouroboros.store import StoreError
from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or
from ouroboros.utils import atomic_write_json, read_json_dict

DEFAULT_UI_PREFERENCES: dict[str, Any] = {
    "widget_order": [],
    # Engine choices are contribution keys; tts_voice is the chosen provider's voice id
    # (empty = the provider's own pick, a voice on this device).
    "voice": {"stt": "", "tts": "", "tts_voice": ""},
    "contributions": {"hidden": [], "order": []},
    # Owner override of a widget card's launch policy, keyed "<skill>:<tab_id>" with a
    # value from the validator's WIDGET_START_MODES. Bounds only: keys are never
    # checked against live widgets, so a temporarily disabled or removed skill keeps
    # the owner's choice instead of losing it with the next discovery.
    "widget_start_mode": {},
    # Owner width of a Widgets card, keyed like widget_start_mode: {"w": masonry columns
    # the card spans, 12 = full width; "h": 0}. ``h`` is reserved for a pinned card height
    # and is stored as 0 until that exists. A POST merges by key and a null value deletes
    # one (the card falls back to its author ``span``); keys are never checked
    # against live widgets. Semantics: docs/DESIGN.md "Widgets board".
    "widget_size": {},
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
    # Main's read cursor, the same monotonic, server-clamped kind against the dialogue store's
    # room revision of chat 1. None until a client first sets it to what Main holds then, as the
    # per-window count this replaced started at zero (DESIGN "Unread").
    "main_seen_revision": None,
    # One message shown in two rooms is read once (DESIGN "Unread"): a client that showed the whole
    # message posts its address (``shared_read``); the host records, per room it counted in (its
    # ``room_revisions`` stamp), that revision as read without moving the cursor, as closed intervals
    # {"<chat_id>": [[from, to], ...]}. Lossless: only that room's own cursor covering a range drops
    # it. Host-owned: a POST naming this key answers 400.
    "read_ahead_revisions": {},
    # Main's empty-state copy, a hidden install-wide preference (no Settings control):
    # never a model reply or a chat-history row.
    "welcome": {"mode": "default", "text": ""},
}
_KNOWN_KEYS = frozenset(DEFAULT_UI_PREFERENCES)
_MAX_WIDGET_ORDER_ITEMS = 200
_MAX_WIDGET_START_MODE_ITEMS = 200
_MAX_WIDGET_KEY_LENGTH = 200
_MAX_WIDGET_SIZE_ITEMS = 200
WIDGET_GRID_COLUMNS = 12  # the bound of w and its full-width value (web/modules/widget_size.js WIDGET_FULL_SPAN)
_SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX = 180, 560
_PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX = 320, 1100
_MAX_PROJECT_CURSORS = 1000
_MAX_PROJECT_ID_LENGTH = 64
_MAX_WELCOME_CHARS = 500
_MAX_SHARED_READS = 64  # addresses one POST may name; recorded facts are never capped


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


def _normalize_widget_size(value: Any) -> dict[str, dict[str, int] | None]:
    """Owner card widths clamped to the board; ``None`` is a key a POST deletes."""
    if not isinstance(value, dict):
        raise ValueError("widget_size must be an object of {widget_key: {w, h}}")
    sizes: dict[str, dict[str, int] | None] = {}
    for widget_key, size in list(value.items())[:_MAX_WIDGET_SIZE_ITEMS]:
        key = str(widget_key or "").strip()
        if not key or len(key) > _MAX_WIDGET_KEY_LENGTH:
            continue
        if size is None:
            sizes[key] = None
            continue
        w, h = (size.get("w"), size.get("h", 0)) if isinstance(size, dict) else (None, None)
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in (w, h)):
            raise ValueError("widget_size values must be null or objects of integer w and h")
        sizes[key] = {"w": max(1, min(WIDGET_GRID_COLUMNS, w)), "h": 0}
    return sizes


def _normalize_preferences(
    raw: dict[str, Any] | None,
    *,
    fill_defaults: bool = True,
) -> dict[str, Any]:
    prefs = copy.deepcopy(DEFAULT_UI_PREFERENCES) if fill_defaults else {}
    if not isinstance(raw, dict):
        return prefs
    for group, keys in (("voice", ("stt", "tts", "tts_voice")), ("contributions", ("hidden", "order"))):
        if group not in raw:
            continue
        value = raw[group]
        if not isinstance(value, dict) or any(key not in keys for key in value):
            raise ValueError(f"{group} must be an object using only {', '.join(keys)}")
        cleaned = dict(prefs.get(group, {}))
        for key, item in value.items():
            if group == "contributions":
                if not isinstance(item, list) or len(item) > 64:
                    raise ValueError(f"{group}.{key} must be a list of at most 64 keys")
                values = item
            else:
                values = [item]
            if any(not isinstance(v, str) or len(v) > 120 for v in values):
                raise ValueError(f"{group}.{key} keys must be strings of at most 120 characters")
            try:
                for v in values:
                    v.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"{group}.{key} keys must be valid Unicode") from None
            cleaned[key] = list(dict.fromkeys(values)) if group == "contributions" else item
        prefs[group] = cleaned
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
    if "widget_size" in raw:
        value = raw.get("widget_size")
        prefs["widget_size"] = {} if value is None else _normalize_widget_size(value)
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
    if "main_seen_revision" in raw:
        value = raw.get("main_seen_revision")
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("main_seen_revision must be an integer or null")
        prefs["main_seen_revision"] = None if value is None else max(0, value)
    if "read_ahead_revisions" in raw:  # stored state only: the POST handler refuses it as input
        value = raw["read_ahead_revisions"]
        if not isinstance(value, dict):
            raise ValueError("read_ahead_revisions must be an object of {chat_id: [[from, to], ...]}")
        prefs["read_ahead_revisions"] = {str(chat): spans for chat, item in value.items()
                                         if str(chat).lstrip("-").isdigit() and (spans := _merged(item))}
    if "welcome" in raw:
        value = raw["welcome"]
        if not isinstance(value, dict) or set(value) != {"mode", "text"}:
            raise ValueError("welcome must have exactly mode and text")
        mode, text = value["mode"], value["text"]
        if mode not in ("default", "hidden", "custom") or not isinstance(text, str):
            raise ValueError("welcome mode or text is invalid")
        if len(text) > _MAX_WELCOME_CHARS:
            raise ValueError("welcome text must be at most 500 characters")
        if mode == "custom" and not text.strip():
            raise ValueError("welcome custom text must be nonblank")
        try:
            # A parsed lone surrogate ("\ud800") is a str but not UTF-8: refused here,
            # the response and the save would fail only after the merge.
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("welcome text must be valid Unicode") from None
        prefs["welcome"] = {"mode": mode, "text": text}
    return prefs


def _merged(intervals: Any) -> list[list[int]]:
    """Closed revision intervals, sorted and merged (adjacent ones join); malformed entries drop."""
    spans = sorted((int(lo), int(hi)) for lo, hi in (item for item in intervals or () if isinstance(item, (list, tuple))
                                                      and len(item) == 2 and all(type(v) is int and v > 0 for v in item))
                   if lo <= hi)
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def _room_cursors(drive_root: Any, prefs: dict) -> dict[int, tuple[int, int]]:
    """``{chat_id: (read cursor, room revision)}`` for Main and every reserved Project."""
    from ouroboros import dialogue_store
    from ouroboros.projects_registry import list_reserved_projects

    revisions = {room["chat_id"]: int(room.get("revision") or 0) for room in dialogue_store.rooms(drive_root)}
    seen = prefs.get("project_seen_revision") or {}
    cursors = {1: (int(prefs.get("main_seen_revision") or 0), revisions.get(1, 0))}
    for project in list_reserved_projects(drive_root):
        chat = int(project.get("chat_id") or 0)
        cursors[chat] = (int(seen.get(str(project.get("id") or "")) or 0), revisions.get(chat, 0))
    return cursors


def _record_shared_reads(drive_root: Any, prefs: dict, addresses: list[str]) -> None:
    """Add each wholly shown message's room revisions above that room's cursor, then drop what
    each room's own cursor now covers. A room the store or registry does not name keeps its facts."""
    from ouroboros.chat_chain import resolve_row

    cursors = _room_cursors(drive_root, prefs)
    ahead = {key: _merged(value) for key, value in (prefs.get("read_ahead_revisions") or {}).items()}
    for address in addresses:
        row, _resolution = resolve_row(drive_root, address)  # a store that cannot answer raises: 503
        stamp = (row or {}).get("room_revisions")
        for chat, revision in (stamp.items() if isinstance(stamp, dict) else ()):
            cursor, held = cursors.get(int(chat), (0, 0)) if str(chat).lstrip("-").isdigit() else (0, 0)
            if type(revision) is int and cursor < revision <= held:
                ahead[str(chat)] = _merged([*ahead.get(str(chat), []), [revision, revision]])
    for chat, spans in list(ahead.items()):
        cursor = cursors.get(int(chat), (0, 0))[0] if chat.lstrip("-").isdigit() else 0
        ahead[chat] = [[max(lo, cursor + 1), hi] for lo, hi in spans if hi > cursor]
        if not ahead[chat]:
            del ahead[chat]
    prefs["read_ahead_revisions"] = ahead


def read_ahead_spans(drive_root: Any, chat_id: Any) -> list[list[int]]:
    """One room's recorded read-ahead revisions (``read_ahead_revisions``); none when unreadable."""
    try:
        stored = _stored_preferences(pathlib.Path(drive_root) / "state" / "ui_preferences.json")
    except Exception:
        return []
    return list(stored.get("read_ahead_revisions", {}).get(str(chat_id), []))


def main_room_revision(drive_root: Any) -> int | None:
    """Main's room revision in the dialogue store (its standalone messages), None while unreadable."""
    try:
        from ouroboros import dialogue_store

        return max(0, int((dialogue_store.room(drive_root, 1) or {}).get("revision") or 0))
    except Exception:
        return None


def _stored_preferences(path: pathlib.Path) -> dict[str, Any]:
    """Saved preferences. ``welcome`` has no Settings control and is edited by hand, so
    a value the POST contract would refuse reads as the default instead of taking every
    other key down with it; the next write stores that default."""
    stored = read_json_dict(path)
    if stored is not None and "welcome" in stored:
        try:
            _normalize_preferences({"welcome": stored["welcome"]}, fill_defaults=False)
        except ValueError:
            stored = {key: value for key, value in stored.items() if key != "welcome"}
    return _normalize_preferences(stored)


async def api_ui_preferences_get(request: Request) -> JSONResponse:
    drive_root = request_drive_root(request)
    path = pathlib.Path(drive_root) / "state" / "ui_preferences.json"
    try:
        prefs = _stored_preferences(path)
        return JSONResponse(prefs)
    except Exception:
        return JSONResponse(dict(DEFAULT_UI_PREFERENCES))


async def api_ui_preferences_post(request: Request) -> JSONResponse:
    body = await request_json_or(request, None)
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    shared = body.pop("shared_read", None)
    unknown = sorted(set(body) - _KNOWN_KEYS)
    if unknown:
        return json_error(f"unknown ui preference key: {unknown[0]}", 400)
    if "read_ahead_revisions" in body:
        return json_error("read_ahead_revisions is recorded by the host; post shared_read addresses", 400)
    if shared is not None and (not isinstance(shared, list) or len(shared) > _MAX_SHARED_READS
                               or any(not isinstance(item, str) or not item.startswith("row:") for item in shared)):
        return json_error(f"shared_read must be a list of at most {_MAX_SHARED_READS} row addresses", 400)
    drive_root = request_drive_root(request)
    path = pathlib.Path(drive_root) / "state" / "ui_preferences.json"
    try:
        with _preferences_lock(path):
            prefs = _stored_preferences(path)
            incoming = _normalize_preferences(body, fill_defaults=False)
            for group in ("voice", "contributions"):
                if group in incoming:
                    prefs[group] = {**prefs[group], **incoming.pop(group)}
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
            if "main_seen_revision" in incoming:
                requested = incoming.pop("main_seen_revision")
                current = main_room_revision(drive_root) if requested is not None else None
                if current is not None:
                    # Never past what Main holds, never backwards; an unreadable store moves nothing.
                    stored = prefs.get("main_seen_revision")
                    prefs["main_seen_revision"] = max(int(stored or 0), min(requested, current))
            if "widget_size" in incoming:
                # Merge by card: a write names only the cards it changes, a null value
                # deletes one, a null map clears them all. A changed key moves last,
                # so the bound keeps the most recently sized cards.
                sizes = {} if body.get("widget_size") is None else {
                    key: size for key, size in prefs["widget_size"].items() if size is not None
                }
                for key, size in incoming.pop("widget_size").items():
                    sizes.pop(key, None)
                    if size is not None:
                        sizes[key] = size
                prefs["widget_size"] = dict(list(sizes.items())[-_MAX_WIDGET_SIZE_ITEMS:])
            prefs.update(incoming)
            if shared:
                _record_shared_reads(drive_root, prefs, list(dict.fromkeys(shared)))
            elif prefs.get("read_ahead_revisions") and ("project_seen_revision" in body or "main_seen_revision" in body):
                try:  # a cursor ACK drops what it now covers; an unreadable store or registry keeps every read fact
                    _record_shared_reads(drive_root, prefs, [])
                except (StoreError, OSError, ValueError):
                    pass
            atomic_write_json(path, prefs, trailing_newline=True)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except TimeoutError as exc:
        return json_error(str(exc), 503)
    except StoreError as exc:
        return json_error(f"dialogue history unavailable: {exc}", 503)
    return JSONResponse({"ok": True, **prefs})


__all__ = [
    "DEFAULT_UI_PREFERENCES",
    "api_ui_preferences_get",
    "api_ui_preferences_post",
    "main_room_revision",
]
