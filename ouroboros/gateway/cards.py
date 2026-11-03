"""``/api/cards/*``: read a living-message card and its app, keep a card on the Widgets page.

The reads are passive. App bytes are served as ``text/plain`` with ``nosniff`` and
``no-store``: the page builds an opaque sandboxed ``srcdoc`` from them, and serving
model-written HTML as a page would run it on the core's own origin (task artifacts
served as ``text/html`` carry no sandbox CSP). Pins are data, kept without review;
the Widgets page lists them through ``pinned_widget_tabs`` (``GET /api/widgets``).
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ouroboros.card_store import CardStoreError, load_card, pin_card, pinned_cards, read_app_bytes, \
    submissions_summary, unpin_card
from ouroboros.gateway._helpers import json_error, request_drive_root

# The reserved owner of pinned cards on the Widgets page; ``canonical_skill_name`` strips
# a leading underscore, so no installed skill can carry this name.
PINNED_CARDS_SKILL = "_cards"


def card_view(root: Any, record: Dict[str, Any]) -> Dict[str, Any]:
    """``CardResponse`` for one stored card."""
    card_id = str(record["card_id"])
    view: Dict[str, Any] = {key: record.get(key) for key in (
        "card_id", "kind", "title", "fallback_text", "chat_id", "task_id", "revision", "created_at", "updated_at")}
    view["pinned"] = any(row.get("card_id") == card_id for row in pinned_cards(root))
    for key in ("components", "data", "widget"):
        if key in record:
            view[key] = record[key]
    if isinstance(record.get("app"), dict):
        view["app"] = {**record["app"], "url": f"/api/cards/{quote(card_id, safe='')}/app"}
    view["submissions_summary"] = submissions_summary(record)
    view["submissions_dropped"] = int(record.get("submissions_dropped") or 0)
    return view


def pinned_widget_tabs(root: Any) -> List[Dict[str, Any]]:
    """Pinned component cards as ``WidgetTab`` rows, after the skills' own widgets."""
    return [{
        "key": f"{PINNED_CARDS_SKILL}:{row['card_id']}", "skill": PINNED_CARDS_SKILL, "tab_id": row["card_id"],
        "title": str(row.get("title") or ""), "icon": "extension", "ws_prefix": "",
        "render": {"kind": "declarative", "schema_version": 1, "start": "auto", "components": row.get("components") or [],
                   **({"data": row["data"]} if isinstance(row.get("data"), dict) else {})},
        "span": 1, "grid_span": 1, "revision": str(row.get("revision") or 1), "pinned_card": True,
    } for row in pinned_cards(root)]


async def api_card_get(request: Request) -> JSONResponse:
    """GET /api/cards/{card_id} — the card's current body and the owner's presses."""
    root = request_drive_root(request)
    record = load_card(root, request.path_params["card_id"])
    if record is None:
        return json_error("card not found", 404, reason_code="card_not_found")
    return JSONResponse(card_view(root, record), headers={"Cache-Control": "no-store"})


async def api_card_app(request: Request) -> Response:
    """GET /api/cards/{card_id}/app — the app's HTML as inert text."""
    root = request_drive_root(request)
    record = load_card(root, request.path_params["card_id"])
    app = record.get("app") if isinstance(record, dict) else None
    data = read_app_bytes(root, app.get("sha256")) if isinstance(app, dict) else None
    if data is None:
        return json_error("card app not found", 404, reason_code="card_app_not_found")
    return Response(data, media_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


async def _widgets_changed(action: str, card_id: str) -> None:
    from ouroboros.gateway.ws import broadcast_ws

    await broadcast_ws({"type": "extension_lifecycle", "skill": PINNED_CARDS_SKILL, "action": action,
                        "data": {"card_id": card_id}})


async def api_card_pin(request: Request) -> JSONResponse:
    """POST /api/cards/{card_id}/pin keeps a component card on Widgets; DELETE removes it."""
    root: pathlib.Path = request_drive_root(request)
    card_id = str(request.path_params["card_id"])
    try:
        if request.method == "DELETE":
            removed = unpin_card(root, card_id)
            if removed:
                await _widgets_changed("card_unpinned", card_id)
            return JSONResponse({"ok": True, "card_id": card_id, "pinned": False})
        pin = pin_card(root, card_id)
    except CardStoreError as exc:
        return json_error(str(exc), 404 if exc.code == "card_not_found" else 409, reason_code=exc.code)
    except (OSError, TimeoutError) as exc:
        return json_error(f"the pin could not be saved: {exc}", 503, reason_code="pin_write_failed")
    await _widgets_changed("card_pinned", card_id)
    return JSONResponse({"ok": True, "card_id": card_id, "pinned": True, "pinned_at": pin["pinned_at"]})


def card_routes() -> list[Route]:
    return [
        Route("/api/cards/{card_id}", endpoint=api_card_get, methods=["GET"]),
        Route("/api/cards/{card_id}/app", endpoint=api_card_app, methods=["GET"]),
        Route("/api/cards/{card_id}/pin", endpoint=api_card_pin, methods=["POST", "DELETE"]),
    ]


__all__ = ["PINNED_CARDS_SKILL", "card_routes", "card_view", "pinned_widget_tabs"]
