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
        if kind == "group":  # the only containers the renderer draws: a group's components and each tab's
            yield from _nodes(node.get("components"), f"{here}.components", aimed)
        for tab_idx, tab in enumerate(node.get("tabs") if kind == "tabs" and isinstance(node.get("tabs"), list) else []):
            if isinstance(tab, dict):
                yield from _nodes(tab.get("components"), f"{here}.tabs[{tab_idx}].components", aimed)


def _media_nodes(components: Any) -> Iterator[Tuple[str, Dict[str, Any], str, bool]]:
    """``_nodes`` narrowed to every image, audio and video, a gallery's listed items included."""
    for here, node, kind, plain in _nodes(components):
        if kind in ("image", "audio", "video"):
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


def _data_value(data: Any, path: str) -> Any:
    """What a component's ``path`` reads from the card's ``data``; None where the renderer falls back to ``src``."""
    current = data
    for part in (part for part in path.split(".") if part):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            index = int(part) if part.isascii() and part.isdigit() and str(int(part)) == part else -1
            current = len(current) if part == "length" else current[index] if 0 <= index < len(current) else None
        else:
            return None
    return current


def _js_text(value: Any) -> str:
    """JavaScript's ``String(value)`` for what JSON holds (a list joins its items with commas, null as nothing)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else repr(value)
    if isinstance(value, list):
        return ",".join("" if item is None else _js_text(item) for item in value)
    return "" if value is None else "[object Object]"


def _draws(node: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """Whether the card renderer finds a source for this media component (``safeMediaSrc``)."""
    path = node.get("path") if isinstance(node.get("path"), str) else ""
    value = _data_value(data, path) if path else None
    return _DRAWABLE_MEDIA.fullmatch(_js_text((node.get("src") or "") if value is None else value).strip(_JS_SPACE)) is not None


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
    for where, node, kind, plain in _nodes(components):
        # A gallery without listed items draws the list its path reads from data, each item as a component.
        if kind != "gallery" or not plain or node.get("items") or isinstance(node.get("items"), list):
            continue
        key = next((node[k] for k in ("path", "items_key") if isinstance(node.get(k), str) and node[k]), "")
        read = _data_value(data, key) if key else data
        for idx, item in enumerate(read if isinstance(read, list) else []):
            kind = str(item.get("type") or "image") if isinstance(item, dict) else "image"
            if kind in ("image", "audio", "video") and not (
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
    named = named_delivery_receipt(ctx, event, mode, "card")
    identity = {"card_id": record["card_id"], "revision": int(record["revision"])}
    if mode in (UNDELIVERED, UNSUPPORTED_PEER):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=json.dumps(
            {**identity, "delivery": named}, ensure_ascii=False)))
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=json.dumps(
        {**identity, "delivery": named, **({} if mode == "live" else {"queued": True})}, ensure_ascii=False)))


__all__ = ["SHOW_CARD_TOOL_SCHEMA", "_show_card"]
