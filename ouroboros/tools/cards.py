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

import json
import pathlib
from typing import Any, Dict, Optional

from ouroboros.tools.arg_feedback import argument_refusal
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

_MAX_TITLE_CHARS = 120
_MAX_FALLBACK_CHARS = 4000
_MAX_BODY_BYTES = 256 * 1024  # a components tree with its data, or a widget's input
_NOT_SHOWN = "The card was not shown."

SHOW_CARD_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "show_card",
    "description": (
        "Put an interactive card into your own message in the owner's chat. Use it when a control "
        "saves the owner work a text answer would leave to them: a choice with details, a small form, "
        "a compact dashboard or checklist, a calculator or simulation. Keep ordinary answers as text.\n\n"
        "Exactly one body. components: a declarative tree the host draws (markdown, metric, callout, kv, "
        "key_value, table, chart, progress, code, json, status, group, tabs, map, calendar, kanban, image, "
        "audio, video, gallery, action, form; no routes); each action and form binds submit: {id, once?} "
        "and data fills the path-reading components. widget: {skill, tab_id, input?} shows an installed "
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


def _card_body(ctx: ToolContext, components: Any, data: Any, widget: Any, app: Any) -> tuple[str, Dict[str, Any], bool]:
    """``(kind, body, interactive)`` for the one supplied body; ValueError names the problem in it."""
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.extension_ui_validation import validate_card_components

    supplied = [name for name, value in (("components", components), ("widget", widget), ("app", app))
                if value not in (None, [], {}, "")]
    if len(supplied) != 1:
        raise ValueError(f"give exactly one body (components, widget or app); got {supplied or 'none'}")
    if data is not None and supplied != ["components"]:
        raise ValueError("data belongs to a components body")
    if components not in (None, [], {}, ""):
        if data is not None and not isinstance(data, dict):
            raise ValueError("data must be an object")
        try:
            checked = validate_card_components(components)
        except ExtensionRegistrationError as exc:
            raise ValueError(str(exc)) from None
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
    if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > _MAX_BODY_BYTES:
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
    from ouroboros.tools.owner_delivery import deliver_owner_event

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
    mode = deliver_owner_event(ctx, {
        "type": "send_card", "chat_id": int(chat_id), "card_id": record["card_id"], "title": record["title"],
        "fallback_text": record["fallback_text"], "kind": kind, "revision": int(record["revision"]),
        "replace": bool(replace), "interactive": interactive,
    })
    receipt = {"card_id": record["card_id"], "revision": int(record["revision"]), "shown": mode == "live",
               **({} if mode == "live" else {"queued": True})}
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=json.dumps(receipt)))


__all__ = ["SHOW_CARD_TOOL_SCHEMA", "_show_card"]
