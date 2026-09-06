"""GET /api/widgets — the Widgets page card list, projected from the live loader.

Built purely from the in-memory extension state: no skill discovery, no
stale-review reconcile, no schedule sync, no disk hashing, no writes
(DEVELOPMENT.md "Passive GET"). ``revision`` is the owning skill's live loader
``content_hash`` — a revision FACT for the page's change signature, not an
ETag and not a cache-busting token.

This module also homes the Widgets TypedDicts that ``gateway/contracts.py``
re-exports, so it imports no transport at module level (Starlette names are
type-only here and the response class is imported inside the handler).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, TypedDict

if TYPE_CHECKING:  # type-only: keeps ``import ouroboros.contracts.api_v1`` transport-free
    from starlette.requests import Request
    from starlette.responses import JSONResponse


class WidgetTab(TypedDict):
    """One Widgets card as served by ``GET /api/widgets``."""

    key: str
    skill: str
    tab_id: str
    title: str
    icon: str
    ws_prefix: str
    render: Dict[str, Any]
    span: int
    grid_span: int
    revision: str


class WidgetsResponse(TypedDict):
    ui_tabs: List[WidgetTab]


class ExtensionLiveSnapshot(TypedDict):
    """``extension_loader.snapshot()`` — the ``live`` block of ``GET /api/extensions``.

    Homed beside the Widgets projection that consumes it so ``gateway/contracts.py``
    stays within its module size band.
    """

    extensions: List[str]
    tools: List[str]
    routes: List[str]
    ws_handlers: List[str]
    ui_tabs: List[Dict[str, Any]]
    settings_sections: List[Dict[str, Any]]


def widget_tabs() -> List[WidgetTab]:
    """Project live UI tabs into Widgets cards stamped with the owning skill's revision.

    One loader read under one lock (``live_widget_projection``): a tab and its
    ``revision`` always come from the same live generation.
    """
    from ouroboros.extension_loader import live_widget_projection

    tabs: List[WidgetTab] = []
    for row in live_widget_projection() or []:
        tab = row["tab"]
        # The TypedDict IS the projection: every declared key except the stamped
        # revision comes straight from the live tab (frame geometry stays in
        # ``render``, which is where the page reads it).
        card: Dict[str, Any] = {
            name: tab.get(name) for name in WidgetTab.__annotations__ if name != "revision"
        }
        card["revision"] = row["revision"]
        tabs.append(card)  # type: ignore[arg-type]
    return tabs


async def api_widgets(_request: Request) -> JSONResponse:
    """GET /api/widgets — live widget cards from the loader only; uncached like the module read."""
    from starlette.responses import JSONResponse

    return JSONResponse({"ui_tabs": widget_tabs()}, headers={"Cache-Control": "no-store"})


__all__ = [
    "ExtensionLiveSnapshot",
    "WidgetTab",
    "WidgetsResponse",
    "api_widgets",
    "widget_tabs",
]
