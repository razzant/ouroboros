"""``show_card``: Ouroboros puts a living, interactive card into its own message.

Exactly one body per card: a declarative ``components`` tree the host draws (the
CARD mode of ``extension_ui_validation``), an installed LIVE skill's ``widget``
shown inline by reference, or a self-contained HTML ``app`` that runs in an opaque
sandboxed frame whose policy blocks fetch and loads (``docs/WIDGET_HOST_PROTOCOL.md`` §9
names what no page policy covers). The record is stored first (``card_store``); the
``send_card`` event then publishes the chat row (``supervisor/card_delivery.py``).
A press on a host-drawn control is the owner's act and returns to the authoring
task as ``card_input``; an app's result lands in the owner's draft and is sent
only by the owner's Send. ``core.get_tools`` registers the entry.
"""

from __future__ import annotations

import base64
import copy
import json
import pathlib
import re
from typing import Any, Dict, Iterator, Optional, Tuple

from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

_MAX_TITLE_CHARS = 120
_MAX_FALLBACK_CHARS = 4000
_MAX_BODY_BYTES = 256 * 1024  # a components tree with its data, or a widget's input
_NOT_SHOWN = "The card was not shown."
# The images one card reads from files, together: the card record and a Widgets pin carry them inline.
_MAX_CARD_IMAGE_BYTES = 2 * 1024 * 1024
# The only media a card draws (``safeMediaSrc`` in web/modules/widget_render.js): an inline data: URI.
_MEDIA_TYPES = ("image", "audio", "video")
_DRAWABLE_MEDIA = re.compile(
    r"data:(image/(?:png|jpeg|jpg|gif|webp)|audio/(?:mpeg|wav|ogg)|video/(?:mp4|webm|ogg));base64,[A-Za-z0-9+/=]+", re.I | re.A)

SHOW_CARD_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "show_card",
    "description": (
        "Put an interactive card into your own message in this work's room (Presence-bound: local, not the "
        "place conversation). Use it when a control "
        "saves the owner work a text answer would leave to them: a choice with details, a small form, "
        "a compact dashboard or checklist, a calculator or simulation. Keep ordinary answers as text.\n\n"
        "Exactly one body. components: a declarative tree the host draws (markdown, metric, callout, kv, "
        "key_value, table, chart, progress, code, json, status, group, tabs, map, calendar, kanban, image, "
        "audio, video, gallery, action, form; no routes); each action and form binds submit: {id, once?} "
        "and data fills the path-reading components. An image takes file: the path of a PNG, JPEG, GIF or "
        "WebP you wrote (absolute, or relative to your workspace; a card's images at most 2 MiB together), "
        "which the host copies into the card, or src: an inline data: URI, as audio and video do. widget: {skill, tab_id, input?} shows an installed "
        "live skill's widget inline. app: {path} or {html} of a self-contained HTML page (at most 512 KiB) "
        "that runs sandboxed: it cannot fetch or load resources, and loading another page closes it. On most "
        "devices it shares the owner's page thread: keep heavy computation in a blob: Worker, never a long "
        "synchronous loop.\n\n"
        "A press on an action or form comes back to you as a [Card input] message with its values (after "
        "your task ended, as the owner's message in this chat: the pressed label and values, as a reply to "
        "the card); once: true accepts only the first press. "
        "An app's submit only fills the owner's draft, which the owner sends. fallback_text is the plain "
        "version for Telegram, exports and any surface that cannot draw the card. replace=<card_id> "
        "updates an earlier card of this chat in place."
    ),
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string", "description": "Card heading (at most 120 characters)."},
        "fallback_text": {"type": "string", "description": (
            "The card's content as plain text (at most 4000 characters), shown wherever the card cannot render.")},
        "components": {"type": "array", "items": {"type": "object"}, "description": (
            "Declarative component tree; action/form use submit: {id: [a-z0-9_]{1,40}, once?: boolean}.")},
        "data": {"type": "object", "description": "Values the components' path fields read (components only)."},
        "widget": {"type": "object", "properties": {
            "skill": {"type": "string"}, "tab_id": {"type": "string"},
            "input": {"type": "object", "description": "Optional input the widget receives."},
        }, "required": ["skill", "tab_id"]},
        "app": {"type": "object", "properties": {
            "path": {"type": "string", "description": "An HTML file you wrote (absolute, or relative to your workspace)."},
            "html": {"type": "string", "description": "The page itself, instead of path."},
        }},
        "replace": {"type": "string", "description": "card_id of an earlier card of this chat to update in place."},
    }, "required": ["title", "fallback_text"]},
}


def _app_bytes(ctx: ToolContext, app: Dict[str, Any]) -> bytes:
    """The app's HTML bytes from exactly one of ``path`` / ``html``; ValueError names the problem."""
    from ouroboros.card_store import MAX_APP_BYTES

    path, html = app.get("path"), app.get("html")
    if bool(path) == bool(html) or not isinstance(path or html, str):
        raise ValueError("app needs exactly one of path or html, as text")
    if html:
        data = html.encode("utf-8")
    else:
        file = pathlib.Path(path).expanduser()
        file = file if file.is_absolute() else pathlib.Path(ctx.active_repo_dir()) / file
        if not file.is_file():
            raise ValueError(f"app file not found: {path}")
        if file.stat().st_size > MAX_APP_BYTES:
            raise ValueError(f"app file is {file.stat().st_size} bytes (limit {MAX_APP_BYTES})")
        data = file.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("app file is not UTF-8 HTML text") from None
    if len(data) > MAX_APP_BYTES:
        raise ValueError(f"app is {len(data)} bytes (limit {MAX_APP_BYTES})")
    return data


def _live_widget_problem(skill: str, tab_id: str) -> str:
    """Empty when ``skill`` has a live widget tab ``tab_id`` in this process's loader."""
    from ouroboros.extension_loader import live_widget_projection

    rows = live_widget_projection(skill)
    if rows is None:
        return f"skill {skill!r} is not live (not installed, disabled, unreviewed or failed to load)"
    tabs = sorted(str(row["tab"].get("tab_id") or "") for row in rows)
    if tab_id not in tabs:
        return f"skill {skill!r} has no live widget tab {tab_id!r} (live tabs: {', '.join(tabs) or 'none'})"
    return ""


def _component_groups(node: Dict[str, Any], kind: str, where: str) -> list[tuple[str, Any]]:
    """The child lists of a group or tabs, at their positions in the declared card."""
    if kind == "group":
        return [(f"{where}.components", node.get("components"))]
    if kind == "tabs" and isinstance(tabs := node.get("tabs"), list):
        return [(f"{where}.tabs[{idx}].components", tab.get("components") if isinstance(tab, dict) else None)
                for idx, tab in enumerate(tabs)]
    return []


def _nodes(components: Any, where: str = "components",
           targeted: bool = False) -> Iterator[Tuple[str, Dict[str, Any], str, bool]]:
    """``(path, node, type, plain)`` of every component of the tree.

    ``plain`` is False for a component the card may lawfully not draw from its own ``data``: one that names a
    ``target`` or is shown on a ``condition_key``, or sits under a component that does.
    """
    for idx, node in enumerate(components if isinstance(components, list) else []):
        if not isinstance(node, dict):
            continue
        here, aimed = f"{where}[{idx}]", targeted or bool(node.get("target")) or bool(node.get("condition_key"))
        kind = str(node.get("type") or "")
        yield here, node, kind, not aimed
        for child_where, children in _component_groups(node, kind, here):
            yield from _nodes(children, child_where, aimed)


def _media_nodes(components: Any) -> Iterator[Tuple[str, Dict[str, Any], str, bool]]:
    """``_nodes`` narrowed to every image, audio and video, a gallery's listed items included."""
    for here, node, kind, plain in _nodes(components):
        if kind in _MEDIA_TYPES:
            yield here, node, kind, plain
        for item_idx, item in enumerate(node.get("items") if kind == "gallery" and isinstance(node.get("items"), list) else []):
            if isinstance(item, dict):
                yield (f"{here}.items[{item_idx}]", item, str(item.get("type") or "image"),
                       plain and not item.get("target") and not item.get("condition_key"))


def _inline_image_files(ctx: ToolContext, components: Any) -> Tuple[Any, int]:
    """The tree with each image's ``file`` read into its inline ``src``, and how many characters that added."""
    from ouroboros.tools.core_artifacts import _detect_image_mime

    if not any("file" in node for _, node, _, _ in _media_nodes(components)):
        return components, 0
    tree, left, added = copy.deepcopy(components), _MAX_CARD_IMAGE_BYTES, 0
    for where, node, kind, _ in _media_nodes(tree):
        if "file" not in node:
            continue
        name = node.pop("file")
        if kind != "image":
            raise ValueError(f"{where}.file: only an image is read from a file; give {kind} as src, an inline data: URI")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}.file must be the path of an image file")
        if name.lstrip().lower().startswith("data:"):
            raise ValueError(f"{where}.file takes a file path; an inline data: URI goes in {where}.src")
        file = pathlib.Path(name).expanduser()
        file = file if file.is_absolute() else pathlib.Path(ctx.active_repo_dir()) / file
        try:
            if not file.is_file():
                looked = "" if pathlib.Path(name).expanduser().is_absolute() else f" (looked for {file})"
                raise ValueError(f"{where}.file not found: {name}{looked}")
            if file.stat().st_size > left:
                raise ValueError(f"{where}.file is {file.stat().st_size} bytes; a card's images take at most "
                                 f"{_MAX_CARD_IMAGE_BYTES} bytes together (send_photo sends a larger image)")
            data = file.read_bytes()
        except OSError as exc:
            raise ValueError(f"{where}.file could not be read: {exc}") from None
        mime = _detect_image_mime(data)
        if not mime.startswith("image/"):
            raise ValueError(f"{where}.file is not a PNG, JPEG, GIF or WebP image: {name}")
        left -= len(data)
        node.pop("path", None)  # a path reads `data`; the file is this image's source
        node["src"] = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        added += len(node["src"])
    return tree, added


# The renderer's own reading of a source, mirrored: `getPath` (web/modules/widget_chart.js), then
# `String(value || '').trim()` (`safeMediaSrc`). Where the two could disagree the card is not refused.
_JS_SPACE = "\t\n\x0b\x0c\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_PATH_UNKNOWN = object()
_OBJECT_PROTOTYPE_KEYS = frozenset({
    "constructor", "__proto__", "toString", "toLocaleString", "valueOf", "hasOwnProperty", "isPrototypeOf",
    "propertyIsEnumerable", "__defineGetter__", "__defineSetter__", "__lookupGetter__", "__lookupSetter__",
})
# String keys on JavaScript's Array.prototype. The reader below handles the array's own length and indices.
_ARRAY_PROTOTYPE_KEYS = _OBJECT_PROTOTYPE_KEYS | {
    "at", "concat", "copyWithin", "entries", "every", "fill", "filter", "find", "findIndex", "findLast",
    "findLastIndex", "flat", "flatMap", "forEach", "includes", "indexOf", "join", "keys", "lastIndexOf",
    "map", "pop", "push", "reduce", "reduceRight", "reverse", "shift", "slice", "some", "sort", "splice",
    "toReversed", "toSorted", "toSpliced", "unshift", "values", "with",
}


def _js_truthy(value: Any) -> bool:
    # JSON arrays and objects are truthy in the renderer even when empty.
    return isinstance(value, (list, dict)) or (bool(value) and value == value)


def _media_state(node: Dict[str, Any], data: dict) -> Any:
    target = node.get("target")
    if not _js_truthy(target) or target == "result":
        return data
    return _PATH_UNKNOWN if not isinstance(target, str) or target in _OBJECT_PROTOTYPE_KEYS else None


def _data_value(current: Any, path: Any) -> Any:
    """Read a JSON path as the renderer does, with a gap for non-JSON property lookups."""
    if current is _PATH_UNKNOWN or not isinstance(path, str):
        return _PATH_UNKNOWN
    for part in (part for part in path.split(".") if part):
        if isinstance(current, dict):
            if part not in current and part in _OBJECT_PROTOTYPE_KEYS:
                return _PATH_UNKNOWN
            current = current.get(part)
        elif isinstance(current, list):
            if part == "length":
                current = len(current)
                continue
            if not (part.isascii() and part.isdecimal()):
                return _PATH_UNKNOWN if part in _ARRAY_PROTOTYPE_KEYS else None
            if (part != "0" and part.startswith("0")) or len(part) > len(str(len(current))):
                return None
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def _js_text(value: Any) -> Optional[str]:
    """Text for media matching, or None where JavaScript number conversion cannot be vouched for."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not -(2**53) < value < 2**53:  # JavaScript's safe integer range, without converting a large Python int
            return None
        return str(int(value)) if isinstance(value, int) or value.is_integer() else repr(value)
    if isinstance(value, list):
        parts = [_js_text(item) for item in value]
        return None if any(part is None for part in parts) else ",".join(parts)
    return "" if value is None else "[object Object]"


def _drawn_source(node: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
    """Inline media from the card state or its fallback ``src``. Empty for an invalid source, None for a gap."""
    path = node.get("path")
    value = _data_value(_media_state(node, data), path) if _js_truthy(path) else None
    if value is _PATH_UNKNOWN:
        return None
    text = _js_text(node.get("src") if value is None else value)
    if text is None:
        return None
    text = text.strip(_JS_SPACE)
    return text if _DRAWABLE_MEDIA.fullmatch(text) else ""


def _draws(node: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Allow an uncertain source through admission. Refuse only media known to have no source."""
    return _drawn_source(node, data) != ""


def _gallery_items(node: Dict[str, Any], data: dict) -> tuple[Optional[str], Optional[list]]:
    """The data key (None for inline items) and its entries (None when the read is uncertain)."""
    items = node.get("items")
    if _js_truthy(items):
        return None, items if isinstance(items, list) else []
    key = next((node[k] for k in ("path", "items_key") if _js_truthy(node.get(k))), "")
    read = _data_value(_media_state(node, data), key)
    if read is _PATH_UNKNOWN:
        return "", None
    return key, read if isinstance(read, list) else []


def _data_gallery_items(components: Any, data: Dict[str, Any]) -> Iterator[Tuple[str, int, str, Any, bool]]:
    """``(gallery place, index, key, item, plain)`` for each item a gallery without listed items draws, as a component:
    the list its ``path`` (else ``items_key``) reads from the card's ``data``; aimed at another target, it reads none."""
    for where, node, kind, plain in _nodes(components):
        if kind != "gallery":
            continue
        key, items = _gallery_items(node, data)
        if key is not None:
            for idx, item in enumerate(items or ()):
                yield where, idx, key, item, plain


def _undrawable_media(components: Any, data: Any) -> str:
    """The first always-shown image, audio or video the card could not draw, worded for its author; else empty."""
    data = data if isinstance(data, dict) else {}
    for where, node, kind, plain in _media_nodes(components):
        if not plain or _draws(node, data):
            continue
        path = node.get("path") if isinstance(node.get("path"), str) else ""
        way = "file: the path of a PNG, JPEG, GIF or WebP you wrote, or src: an inline data: URI" if kind == "image" \
            else "src: an inline data: URI (audio mpeg, wav or ogg; video mp4, webm or ogg)"
        note = f"; path {path!r} reads a value of data, not a file" if path else ""
        return f"{where}: this {kind} has no source a card can draw{note}. Give it {way}"
    for where, idx, key, item, plain in _data_gallery_items(components, data):
        kind = str(item.get("type") or "image") if isinstance(item, dict) else "image"
        if plain and kind in _MEDIA_TYPES and not (
                isinstance(item, dict) and (item.get("target") or item.get("condition_key") or _draws(item, data))):
            return (f"{where}: item {idx} of data {key!r} has no source a card can draw. Give each item src: an "
                    f"inline data: URI, or list the images in the gallery's items, where an image takes file")
    return ""


def _card_body(ctx: ToolContext, components: Any, data: Any, widget: Any, app: Any) -> tuple[str, Dict[str, Any], bool]:
    """``(kind, body, interactive)`` for the one supplied body; ValueError names the problem in it."""
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.extension_ui_validation import validate_card_components

    supplied = [name for name, value in (("components", components), ("widget", widget), ("app", app))
                if value not in (None, [], {}, "")]
    inlined = 0  # characters of the images read from files, which the body bound does not count
    if len(supplied) != 1:
        raise ValueError(f"give exactly one body (components, widget or app); got {supplied or 'none'}")
    if data is not None and supplied != ["components"]:
        raise ValueError("data belongs to a components body")
    if components not in (None, [], {}, ""):
        if data is not None and not isinstance(data, dict):
            raise ValueError("data must be an object")
        components, inlined = _inline_image_files(ctx, components)
        try:
            checked = validate_card_components(components)
        except ExtensionRegistrationError as exc:
            raise ValueError(str(exc)) from None
        if problem := _undrawable_media(checked["components"], data):
            raise ValueError(problem)
        body = {"components": checked["components"], **({"data": data} if data is not None else {})}
    elif widget not in (None, {}, ""):
        if not isinstance(widget, dict) or not all(isinstance(widget.get(k), str) and widget.get(k) for k in ("skill", "tab_id")):
            raise ValueError("widget needs skill and tab_id")
        if "input" in widget and not isinstance(widget["input"], dict):
            raise ValueError("widget.input must be an object")
        if problem := _live_widget_problem(widget["skill"], widget["tab_id"]):
            raise ValueError(problem + "; show its content as components instead, or enable the skill first")
        body = {"widget": {key: widget[key] for key in ("skill", "tab_id", "input") if key in widget}}
    else:
        if not isinstance(app, dict):
            raise ValueError("app must be an object with path or html")
        return "app", {"app_bytes": _app_bytes(ctx, app)}, True
    if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) - inlined > _MAX_BODY_BYTES:
        raise ValueError(f"the {supplied[0]} body exceeds {_MAX_BODY_BYTES} bytes as JSON")
    return supplied[0], body, supplied[0] == "widget" or bool(checked["actions"])


def _visible_media(components: Any, data: dict, *, replacing: bool) -> Iterator[tuple[str, Optional[str]]]:
    """Media positions and sources in the card's draw, including groups and galleries read from data.

    The card mount supplies only ``result`` data and no skill route. A new tab set draws its first tab.
    A replacement may reuse the viewer's active tab, which the host cannot attest.
    """
    active: set[int] = set()

    def walk(nodes, position, *, gallery=False, certain=True):
        if not isinstance(nodes, list):
            return
        if id(nodes) in active:
            raise ValueError("card data refers back to its own gallery")
        active.add(id(nodes))
        try:
            for idx, node in enumerate(nodes):
                where = position.format(idx)
                if not isinstance(node, dict):
                    if gallery and node is None:
                        raise ValueError("card data has a null gallery entry")
                    if gallery:
                        yield f"media {where}", None
                    continue
                condition = node.get("condition_key")
                state = _media_state(node, data)
                visible = _data_value({} if state is None else state, condition) if _js_truthy(condition) else True
                if not _js_truthy(visible):
                    continue
                known = certain and visible is not _PATH_UNKNOWN
                raw_type = node.get("type")
                kind = (_js_text(raw_type) or "") if _js_truthy(raw_type) else ("image" if gallery else "")
                if kind in _MEDIA_TYPES:
                    routed = any(_js_truthy(node.get(key)) for key in ("route", "api_route"))
                    yield f"{kind} {where}", _drawn_source(node, data) if known and not routed else None
                elif kind == "gallery":
                    key, items = _gallery_items(node, data)
                    if items is None or (items and _js_truthy(node.get("items_key")) and _js_truthy(node.get("route_prefix"))):
                        yield f"gallery {where}", None
                    else:
                        item_position = f"{where}.items[{{}}]" if key is None else f"{where} item {{}}"
                        yield from walk(items, item_position, gallery=True, certain=known)
                else:
                    children = _component_groups(node, kind, where)
                    if kind == "tabs":
                        if replacing and len(children) > 1:
                            known = False
                        else:
                            children = children[:1]
                    for child_where, rows in children:
                        yield from walk(rows, f"{child_where}[{{}}]", certain=known)
        finally:
            active.remove(id(nodes))

    yield from walk(components, "components[{}]")


def _inline_media_base64(source: str) -> str:
    """Pad a browser-decodable base64 payload for the shared byte receipt, or leave an unreadable one unknown."""
    encoded = source.split(",", 1)[1]
    if len(encoded) % 4 == 0:
        encoded = encoded.removesuffix("=").removesuffix("=")
    if not encoded or len(encoded) % 4 == 1 or "=" in encoded:
        return ""
    return encoded + "=" * (-len(encoded) % 4)


def _shown_media(kind: str, body: Dict[str, Any], app: Any, *, replacing: bool) -> list:
    """Known media bytes and any unresolved media, from the accepted body on its existing show receipt."""
    from ouroboros.tools.owner_delivery import shown_material

    if kind == "app":
        path = app.get("path") if isinstance(app, dict) else None
        return [{"name": pathlib.Path(path).name if path else "app page", **body["app"]}]
    data = body.get("data") if isinstance(body.get("data"), dict) else {}
    material = []
    for name, source in _visible_media(body.get("components"), data, replacing=replacing):
        encoded = _inline_media_base64(source) if source else ""
        fact = shown_material(name, b64=encoded) if encoded else {}
        material.append(fact or {"bytes_unknown": True})
    return material


def _show_card(ctx: ToolContext, title: str = "", fallback_text: str = "", components: Any = None,
               data: Any = None, widget: Any = None, app: Any = None, replace: str = "") -> str:
    """Store the card, then publish it to the owner's chat (live, or with the task's deferred events)."""
    chat_id: Optional[int] = getattr(ctx, "current_chat_id", None)
    if chat_id is None or chat_id == "":
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE",
                                                    text="⚠️ No active chat — cannot show a card."))
    problems = []
    if not isinstance(title, str) or not title.strip() or len(title) > _MAX_TITLE_CHARS:
        problems.append(f"title is required text of at most {_MAX_TITLE_CHARS} characters")
    if not isinstance(fallback_text, str) or not fallback_text.strip() or len(fallback_text) > _MAX_FALLBACK_CHARS:
        problems.append(f"fallback_text is required plain text of at most {_MAX_FALLBACK_CHARS} characters")
    try:
        kind, body, interactive = _card_body(ctx, components, data, widget, app)
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        return argument_refusal(ctx, "SHOW_CARD_ARG_ERROR", problems, effect=_NOT_SHOWN)

    from ouroboros.card_store import CardStoreError, store_app_bytes, write_card
    from ouroboros.projects_registry import project_chat_for_task_tree
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tools.owner_delivery import UNDELIVERED, UNSUPPORTED_PEER, deliver_owner_event, named_delivery_receipt

    metadata = getattr(ctx, "task_metadata", {}) or {}
    root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or metadata.get("budget_drive_root") or ctx.drive_root)
    try:
        if kind == "app":
            body = {"app": store_app_bytes(root, body["app_bytes"])}
        record = write_card(root, chat_id=int(chat_id), task_id=str(getattr(ctx, "task_id", "") or ""),
                            title=title.strip(), fallback_text=fallback_text, kind=kind, body=body,
                            replace=str(replace or ""), same_room=lambda card: project_chat_for_task_tree(
                                canonical_data_root(ctx), card.get("task_id")) == int(chat_id))
    except CardStoreError as exc:
        return argument_refusal(ctx, exc.code, [str(exc)], effect=_NOT_SHOWN)
    except (OSError, TimeoutError) as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="LEGACY_TOOL_ERROR",
                                                    text=f"⚠️ The card could not be stored: {exc}"))
    event = {
        "type": "send_card", "chat_id": int(chat_id), "card_id": record["card_id"], "title": record["title"],
        "fallback_text": record["fallback_text"], "kind": kind, "revision": int(record["revision"]),
        "replace": bool(replace), "interactive": interactive,
    }
    mode = deliver_owner_event(ctx, event)
    named = named_delivery_receipt(ctx, event, mode, "card", lambda: _shown_media(kind, body, app, replacing=bool(replace)))
    identity = {"card_id": record["card_id"], "revision": int(record["revision"])}
    if mode in (UNDELIVERED, UNSUPPORTED_PEER):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=json.dumps(
            {**identity, "delivery": named}, ensure_ascii=False)))
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=json.dumps(
        {**identity, "delivery": named, **({} if mode == "live" else {"queued": True})}, ensure_ascii=False)))


__all__ = ["SHOW_CARD_TOOL_SCHEMA", "_show_card"]
