"""Owner-local UI preference endpoints."""
from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.contracts.chat_id_policy import MAIN_THREAD_ID
from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso

DEFAULT_UI_PREFERENCES: dict[str, Any] = {
    "widget_order": [],
    "nested_subagents_expanded": False,
    # Resizable side sections (0 = use the CSS default). Clamped to sane ranges so
    # a stored value can never collapse or run away with the layout.
    "sidebar_width": 0,
    "project_panel_width": 0,
    # Monotonic, server-clamped read cursors, NESTED per thread since project
    # threads (T1): ``{project_id: {thread_id: revision}}``. A thread is unread
    # exactly when its durable visible_revision is greater than its own value.
    "project_seen_revision": {},
    # Owner drag-and-drop ordering (D3), same shape family as ``widget_order``:
    # an explicit prefix of ids; anything not listed keeps the default order
    # behind it (new project on top, new thread on top within its project).
    "project_order": [],
    "project_thread_order": {},
    # One-minor compatibility inputs: accepted as loud no-ops.
    "project_last_viewed": {},
    "project_hidden": {},
}
_KNOWN_KEYS = frozenset(DEFAULT_UI_PREFERENCES)
_MAX_WIDGET_ORDER_ITEMS = 200
_MAX_WIDGET_KEY_LENGTH = 200
_SIDEBAR_WIDTH_MIN, _SIDEBAR_WIDTH_MAX = 180, 560
_PROJECT_PANEL_WIDTH_MIN, _PROJECT_PANEL_WIDTH_MAX = 320, 1100
_MAX_PROJECT_CURSORS = 1000
# The 64-char key budget was sized for the FLAT cursor, where "thread" could only
# ever have been encoded by suffixing the project id (`pid:12`) — that packing is
# exactly what the nested shape (D1) exists to avoid. Nesting therefore does NOT
# spend the project-id budget: 64 still bounds a project id alone, and thread ids
# are small integers bounded separately by _MAX_THREAD_CURSORS. Revisited and
# deliberately left at 64 (X6).
_MAX_PROJECT_ID_LENGTH = 64
# Per project, PER REQUEST — and ONLY per request. A project's thread count is
# owner-driven, so this must never bound what is STORED; the merge prunes a
# stored lane against the project's live threads instead (see
# api_ui_preferences_post). Enforcing that distinction takes an argument, not a
# comment: `_normalize_seen_revision(..., bound_threads=False)` is what both
# handlers pass for the DISK read, because applying this cap there turned "keep
# the last 200" into "keep the FIRST 200 in stored order" and evicted the lane
# an owner had just acknowledged — on the READ path, where nothing else can put
# it back. Only the request body is bounded here.
_MAX_THREAD_CURSORS = 200
_MAX_THREAD_ID_LENGTH = 12
_MAX_ORDERED_PROJECTS = 1000
_MAX_ORDERED_THREADS = 200
_DEPRECATED_UI_PREFERENCE_EVENTS: set[str] = set()


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


def _normalize_seen_revision(value: Any, *, bound_threads: bool = True) -> dict[str, dict[str, int]]:
    """Normalize the read cursor to its NESTED ``{project: {thread: rev}}`` shape.

    ``bound_threads`` selects the LANE. True is the REQUEST lane: one POST body
    may carry at most ``_MAX_THREAD_CURSORS`` cursors per project. False is the
    STORED lane, where the document has already been bounded by the merge's
    existence prune (dead threads dropped) and re-applying a per-request cap
    would silently discard live cursors in stored order — which is the order a
    just-written ACK sits LAST in.

    BREAKING ABI migration (X6), shipped atomically with the thread UI. The
    compatibility window is the per-entry branch below: a FLAT ``{pid: int}``
    entry — every value stored before this release, and any request from a
    client that predates it — maps to ``{pid: {"0": int}}``. Thread #0 IS the
    project's original chat (``thread_chat_id(pid, 0) == project_chat_id(pid)``),
    so the old number describes exactly that thread and no unread state is
    invented or lost. Mixed input (some projects flat, some nested) is normal
    during the window and is accepted per entry, not per document.

    Rejection stays LOUD (``ValueError`` -> HTTP 400) for anything that is
    neither an int nor an object of ints: a silently-dropped cursor would look
    like a room that refuses to go read.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            "project_seen_revision must be an object of {project_id: {thread_id: revision}}"
        )
    cleaned: dict[str, dict[str, int]] = {}
    for pid, per_thread in list(value.items())[:_MAX_PROJECT_CURSORS]:
        key = str(pid or "").strip()[:_MAX_PROJECT_ID_LENGTH]
        if not key:
            continue
        if isinstance(per_thread, bool):
            # bool is an int subclass; a boolean cursor is a client bug, not a 0/1 revision.
            raise ValueError("project_seen_revision values must be integers")
        if isinstance(per_thread, dict):
            threads: dict[str, int] = {}
            entries = list(per_thread.items())
            if bound_threads:
                entries = entries[:_MAX_THREAD_CURSORS]
            for tid, revision in entries:
                thread_key = str(tid if tid is not None else "").strip()[:_MAX_THREAD_ID_LENGTH]
                if not thread_key or isinstance(revision, bool):
                    if isinstance(revision, bool):
                        raise ValueError("project_seen_revision values must be integers")
                    continue
                try:
                    thread_key = str(int(thread_key))
                except (TypeError, ValueError):
                    raise ValueError("project_seen_revision thread ids must be integers")
                try:
                    threads[thread_key] = max(0, int(revision or 0))
                except (TypeError, ValueError):
                    raise ValueError("project_seen_revision values must be integers")
            cleaned[key] = threads
            continue
        # Compatibility window: a flat per-project number IS thread #0's cursor.
        try:
            cleaned[key] = {str(MAIN_THREAD_ID): max(0, int(per_thread or 0))}
        except (TypeError, ValueError):
            raise ValueError("project_seen_revision values must be integers")
    return cleaned


def _normalize_thread_order(value: Any) -> dict[str, list[str]]:
    """``{project_id: [thread_id, ...]}`` — the owner's manual thread order (D3)."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("project_thread_order must be an object of {project_id: [thread_id]}")
    cleaned: dict[str, list[str]] = {}
    for pid, order in list(value.items())[:_MAX_PROJECT_CURSORS]:
        key = str(pid or "").strip()[:_MAX_PROJECT_ID_LENGTH]
        if not key:
            continue
        if not isinstance(order, list):
            raise ValueError("project_thread_order values must be lists of thread ids")
        ids: list[str] = []
        seen: set[str] = set()
        for item in order[:_MAX_ORDERED_THREADS]:
            if isinstance(item, bool):
                raise ValueError("project_thread_order values must be lists of thread ids")
            try:
                thread_key = str(int(item))
            except (TypeError, ValueError):
                raise ValueError("project_thread_order values must be lists of thread ids")
            if thread_key in seen:
                continue
            seen.add(thread_key)
            ids.append(thread_key)
        cleaned[key] = ids
    return cleaned


def _normalize_preferences(
    raw: dict[str, Any] | None,
    *,
    fill_defaults: bool = True,
    bound_threads: bool = True,
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
    if "project_seen_revision" in raw:
        prefs["project_seen_revision"] = _normalize_seen_revision(
            raw.get("project_seen_revision"), bound_threads=bound_threads
        )
    if "project_order" in raw:
        value = raw.get("project_order")
        if value is None:
            prefs["project_order"] = []
        elif not isinstance(value, list):
            raise ValueError("project_order must be a list of project ids")
        else:
            ordered: list[str] = []
            seen_pids: set[str] = set()
            for item in value[:_MAX_ORDERED_PROJECTS]:
                if not isinstance(item, str):
                    raise ValueError("project_order must be a list of project ids")
                key = item.strip()[:_MAX_PROJECT_ID_LENGTH]
                if not key or key in seen_pids:
                    continue
                seen_pids.add(key)
                ordered.append(key)
            prefs["project_order"] = ordered
    if "project_thread_order" in raw:
        prefs["project_thread_order"] = _normalize_thread_order(raw.get("project_thread_order"))
    for deprecated in ("project_last_viewed", "project_hidden"):
        if deprecated in raw and raw.get(deprecated) is not None and not isinstance(raw.get(deprecated), dict):
            raise ValueError(f"{deprecated} must be an object")
        if deprecated in raw:
            prefs[deprecated] = {}
    return prefs


def _legacy_keys(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return sorted(
        key for key in ("project_hidden", "project_last_viewed")
        if key in raw and isinstance(raw.get(key), dict) and bool(raw.get(key))
    )


def _deprecated_warning(drive_root: Any, keys: list[str], source: str) -> dict | None:
    selected = sorted(set(keys))
    if not selected:
        return None
    warning = {
        "type": "deprecated_ui_preferences_ignored",
        "settings": selected,
        "source": source,
        "replacement": "project_seen_revision",
    }
    event_key = f"{pathlib.Path(drive_root).resolve(strict=False)}:{','.join(selected)}"
    if event_key not in _DEPRECATED_UI_PREFERENCE_EVENTS:
        _DEPRECATED_UI_PREFERENCE_EVENTS.add(event_key)
        try:
            append_jsonl(
                pathlib.Path(drive_root) / "logs" / "events.jsonl",
                {"ts": utc_now_iso(), **warning},
            )
        except Exception:
            # Compatibility warning remains present in the response even when the
            # optional event sink is unavailable.
            pass
    return warning


async def api_ui_preferences_get(request: Request) -> JSONResponse:
    drive_root = request_drive_root(request)
    path = pathlib.Path(drive_root) / "state" / "ui_preferences.json"
    try:
        raw = read_json_dict(path)
        # The STORED lane: reading is not a request, so the per-request cursor cap
        # must not apply. Bounding here evicted whichever lane the document listed
        # last — the one the owner's most recent ACK had just written — so the
        # thread painted unread again, ACK'd again, and was dropped again forever.
        prefs = _normalize_preferences(raw, bound_threads=False)
        warning = _deprecated_warning(drive_root, _legacy_keys(raw), "stored")
        return JSONResponse({**prefs, **({"warnings": [warning]} if warning else {})})
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
    incoming_legacy = _legacy_keys(body)
    try:
        with _preferences_lock(path):
            try:
                # STORED lane (see api_ui_preferences_get): no per-request cap.
                prefs = _normalize_preferences(read_json_dict(path), bound_threads=False)
            except Exception:
                # A stored document this normalizer refuses must not WEDGE the
                # file. Letting its ValueError escape made every POST 400 —
                # including the very write that would have replaced the bad
                # value — so the preferences file became permanently unwritable
                # and no owner action could recover it. Falling back to defaults
                # means an incoming write HEALS it: the refused document is
                # replaced by defaults plus whatever this request carries. Only
                # the DISK read gets this fallback; a bad request body below
                # still 400s, because rejecting bad input is the loud contract.
                prefs = dict(DEFAULT_UI_PREFERENCES)
            incoming = _normalize_preferences(body, fill_defaults=False)
            if "project_seen_revision" in incoming:
                from ouroboros.projects_registry import (
                    get_project,
                    get_reserved_project,
                    project_threads,
                )

                merged = {
                    pid: dict(threads)
                    for pid, threads in (prefs.get("project_seen_revision") or {}).items()
                }
                for project_id, requested in incoming.pop("project_seen_revision").items():
                    project = get_project(drive_root, project_id)
                    if project is None:
                        # `get_project` is ACTIVE-only, so a tombstoned project
                        # used to reach `continue` and keep its whole cursor lane
                        # forever — the project-level twin of the dead-thread lanes
                        # pruned below. Consult the lifecycle-agnostic lookup and
                        # drop the lane once the row is genuinely tombstoned.
                        # `deleting` rows are NOT dropped (the delete can still be
                        # observed, and their threads are still real), and an
                        # unknown id is not ours to touch: both keep `continue`.
                        reserved = get_reserved_project(drive_root, project_id)
                        if reserved is not None and str(reserved.get("lifecycle") or "") == "tombstoned":
                            merged.pop(project_id, None)
                        continue
                    # Clamp EACH thread against its OWN durable revision, read
                    # through the canonical projection so thread #0 is clamped by
                    # `thread0_visible_revision` and not by the project-wide
                    # aggregate (which any sibling thread can advance). Clamping
                    # thread #0 against the aggregate would let a sibling's
                    # message silently mark thread #0 read.
                    ceilings = {
                        str(thread["id"]): max(0, int(thread.get("visible_revision") or 0))
                        for thread in project_threads(project)
                    }
                    # PRUNE by existence, never by insertion order. `ceilings` is
                    # the project's complete live thread set, so a lane key it does
                    # not contain belongs to a thread that no longer exists and is
                    # dropped here — which is also the only bound this lane needs.
                    # Trimming to the _MAX_THREAD_CURSORS newest-inserted entries
                    # instead would, on a project with more threads than that, drop
                    # whichever lane happened to be written FIRST — routinely
                    # thread #0 — and silently re-mark the project's main chat
                    # unread. That constant still bounds one REQUEST
                    # (`_normalize_seen_revision`); it has no business bounding a
                    # lane whose size the owner's own thread count decides.
                    lane = {
                        thread_id: revision
                        for thread_id, revision in (merged.get(project_id) or {}).items()
                        if thread_id in ceilings
                    }
                    for thread_id, revision in requested.items():
                        if thread_id not in ceilings:
                            continue  # unknown/removed thread: never newly admitted
                        acknowledged = min(max(0, int(revision or 0)), ceilings[thread_id])
                        lane[thread_id] = max(int(lane.get(thread_id) or 0), acknowledged)
                    merged[project_id] = lane
                if len(merged) > _MAX_PROJECT_CURSORS:
                    # Last-resort backstop only. The real bound is the tombstone
                    # prune above: a deleted project's lane is dropped when it is
                    # next written to, so delete-churn no longer piles dead lanes
                    # up against this cap. It stays because nothing guarantees a
                    # write ever names a given tombstoned project again — but it
                    # evicts by STORED ORDER, so if it ever fires it can drop a
                    # live project. Reaching it means 1000+ distinct project ids
                    # in one owner's cursor, which the prune makes implausible.
                    merged = dict(list(merged.items())[-_MAX_PROJECT_CURSORS:])
                prefs["project_seen_revision"] = merged
            incoming.pop("project_last_viewed", None)
            incoming.pop("project_hidden", None)
            prefs.update(incoming)
            prefs["project_last_viewed"] = {}
            prefs["project_hidden"] = {}
            atomic_write_json(path, prefs, trailing_newline=True)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except TimeoutError as exc:
        return json_error(str(exc), 503)
    warning = _deprecated_warning(drive_root, incoming_legacy, "incoming")
    return JSONResponse({"ok": True, **prefs, **({"warnings": [warning]} if warning else {})})


__all__ = [
    "DEFAULT_UI_PREFERENCES",
    "api_ui_preferences_get",
    "api_ui_preferences_post",
]
