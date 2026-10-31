"""GET /api/widgets — the Widgets page card list, projected from the live loader.

Built from the in-memory extension state plus the owner's pinned cards (one small
file, ``gateway/cards.py``): no skill discovery, no stale-review reconcile, no
schedule sync, no disk hashing, no writes (DEVELOPMENT.md "Passive GET"). ``revision`` is the owning skill's live loader
``content_hash`` — a revision FACT for the page's change signature, not an
ETag and not a cache-busting token.

``POST /api/widgets/fault`` is the one write beside it: a fault a running module
widget's card already shows (web/modules/widget_module.js) becomes one
``widget_fault`` row in the existing ``logs/events.jsonl``, client-reported and
task-less, bounded per widget per time window. It reads no skill state.

This module also homes the Widgets TypedDicts that ``gateway/contracts.py``
re-exports, so it imports no transport at module level (Starlette names are
type-only here and the response class is imported inside the handler).
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Literal

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

if TYPE_CHECKING:  # type-only: keeps ``import ouroboros.contracts.api_v1`` transport-free
    from starlette.requests import Request
    from starlette.responses import JSONResponse


class WidgetTab(TypedDict):
    """One Widgets card as served by ``GET /api/widgets``. ``pinned_card`` marks a living-message
    card the owner kept (skill ``_cards``, ``gateway/cards.py``); its presses use the ``card:``
    decision family, never a skill route."""

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
    generation: NotRequired[str]
    pinned_card: NotRequired[bool]


class WidgetsResponse(TypedDict):
    ui_tabs: List[WidgetTab]


class WidgetAssetLeaseRequest(TypedDict):
    """``scope`` ``files`` (default): an ESM mount's declared files; ``routes``: GET/HEAD of a classic module or
    route-iframe mount's own skill routes, at ``/api/widget-resources/``. One kind never reads the other.
    ``token`` (routes only) re-establishes the lease the mount's running frame already uses."""
    skill: str
    tab_id: str
    generation: str
    scope: NotRequired[Literal["files", "routes"]]
    token: NotRequired[str]


class WidgetAssetTokenRequest(TypedDict):
    token: str


class WidgetAssetRevokeResponse(TypedDict):
    ok: bool


class WidgetAssetLeaseResponse(TypedDict):
    token: str
    base_url: str
    expires_in: int
    renew_after: int


class WidgetFaultRequest(TypedDict):
    """``POST /api/widgets/fault``: one fault a mounted module widget's card shows. ``skill``, ``tab_id`` and
    ``revision`` are the page's own mount, never the frame's words; ``where`` is the Widgets board or a living
    card in a message; ``message``/``source``/``line`` are the frame's clipped report (``source``: the script
    file, or the blocked URL of a ``csp`` refusal)."""

    skill: str
    tab_id: str
    revision: str
    where: Literal["board", "message"]
    kind: Literal["error", "rejection", "csp"]
    message: str
    source: str
    line: int


class WidgetFaultResponse(TypedDict):
    """``logged`` false: this widget's or every widget's window was full; the widget's next row counts it."""

    ok: bool
    logged: bool


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
    contributions: List[Dict[str, Any]]


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
            name: tab.get(name) for name in WidgetTab.__annotations__ if name not in {"revision", "generation", "pinned_card"}
        }
        card["revision"] = row["revision"]
        card["generation"] = row["generation"]
        tabs.append(card)  # type: ignore[arg-type]
    return tabs


async def api_widgets(request: Request) -> JSONResponse:
    """GET /api/widgets — live widget cards from the loader, then the owner's pinned cards; uncached."""
    from starlette.responses import JSONResponse

    from ouroboros.gateway._helpers import request_drive_root
    from ouroboros.gateway.cards import pinned_widget_tabs

    tabs = [*widget_tabs(), *pinned_widget_tabs(request_drive_root(request))]
    return JSONResponse({"ui_tabs": tabs}, headers={"Cache-Control": "no-store"})


# The frame's own clip (web/modules/widget_frame.js) is the bound: a longer field is not our page's report.
FAULT_MESSAGE_MAX = 500
FAULT_SOURCE_MAX = 200
# Even fully escaped JSON (six bytes per character) must fit the admitted field bounds.
_FAULT_BODY_MAX = 8192
_FAULT_LINE_MAX = 10_000_000
_FAULT_TOKEN = re.compile(r"\w[\w.-]{0,127}")
_FAULT_REVISION = re.compile(r"[\w.:-]{0,128}")
# Rows per widget per window, and for all widgets together, so a remount storm or a spray of names cannot flood
# the log; a window replenishes on the first report after it ends. The least recent widget is forgotten first.
FAULT_WINDOW_SEC = 60.0
FAULT_ROWS_PER_WIDGET = 10
FAULT_ROWS_PER_WINDOW = 100
_FAULT_WIDGETS_MAX = 256
_FAULT_WINDOWS: "OrderedDict[str, List[float]]" = OrderedDict()  # key -> [window start, rows, suppressed]


def _fault_record(body: Dict[str, Any]) -> Dict[str, Any] | None:
    """The fields of a report ``validate_ingress`` admitted, within the frame's bounds, or None."""
    names = (body["skill"], body["tab_id"])
    if not all(_FAULT_TOKEN.fullmatch(name) for name in names) or not _FAULT_REVISION.fullmatch(body["revision"]):
        return None
    if len(body["message"]) > FAULT_MESSAGE_MAX or len(body["source"]) > FAULT_SOURCE_MAX:
        return None
    if not 0 <= body["line"] <= _FAULT_LINE_MAX:
        return None
    record = {name: body[name] for name in WidgetFaultRequest.__annotations__}
    # JS clips UTF-16 units: a boundary can split an emoji. Keep the bounded
    # report, replacing only unpaired surrogates before the UTF-8 log writer.
    from ouroboros.gateway.widget_assets import redact_asset_links

    for name in ("message", "source"):
        record[name] = record[name].encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        record[name] = redact_asset_links(record[name])
    return record


def _fault_admit(widget: str, now: float) -> tuple[bool, int]:
    """Whether this widget may write a row now, and how many of its faults went unwritten before it."""
    admitted = True
    carried = 0
    for key in ("", widget):  # "" is every widget together
        window = _FAULT_WINDOWS.get(key)
        if window is None or now - window[0] >= FAULT_WINDOW_SEC:
            window = [now, 0, window[2] if window else 0]
        _FAULT_WINDOWS[key] = window
        _FAULT_WINDOWS.move_to_end(key)
        if window[1] >= (FAULT_ROWS_PER_WINDOW if key == "" else FAULT_ROWS_PER_WIDGET):
            admitted = False
    widget_window = _FAULT_WINDOWS[widget]
    if admitted:
        _FAULT_WINDOWS[""][1] += 1
        widget_window[1] += 1
        carried, widget_window[2] = int(widget_window[2]), 0
    else:
        widget_window[2] += 1
    while len(_FAULT_WINDOWS) > _FAULT_WIDGETS_MAX + 1:
        oldest = next(key for key in _FAULT_WINDOWS if key != "")
        del _FAULT_WINDOWS[oldest]
    return admitted, carried


async def _bounded_body(request: Request) -> bytes | None:
    """The request body, or None past ``_FAULT_BODY_MAX`` bytes (read no further)."""
    declared = request.headers.get("content-length")
    if declared and (not declared.isdigit() or int(declared) > _FAULT_BODY_MAX):
        return None
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > _FAULT_BODY_MAX:
            return None
    return bytes(chunks)


async def api_widget_fault_post(request: Request) -> JSONResponse:
    """``WidgetFaultRequest`` → ``200 {ok, logged}``; ``400`` another shape, ``413`` a body past 8 KiB, ``503`` the
    row could not be appended. The row has no task: no task's context or card reads it (``GET /api/logs/events``
    does), it never becomes a chat row or a notification, and it is the page's report, not a verified execution."""
    import json

    from starlette.responses import JSONResponse

    from ouroboros.gateway._helpers import json_error, request_drive_root, run_sync_to_completion
    from ouroboros.gateway.schema import validate_ingress
    from ouroboros.observability import redact_projection
    from ouroboros.utils import append_jsonl, utc_now_iso

    raw = await _bounded_body(request)
    if raw is None:
        return json_error(f"a widget fault report is at most {_FAULT_BODY_MAX} bytes", 413)
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    record = None if validate_ingress(body, WidgetFaultRequest) else _fault_record(body)
    if record is None:
        return json_error("expected skill, tab_id, revision, where (board|message), kind (error|rejection|csp), "
                          f"message (<= {FAULT_MESSAGE_MAX}), source (<= {FAULT_SOURCE_MAX}) and a line number", 400)
    admitted, suppressed = _fault_admit(f"{record['skill']}:{record['tab_id']}", time.monotonic())
    if not admitted:
        return JSONResponse({"ok": True, "logged": False})
    row = {"ts": utc_now_iso(), "type": "widget_fault", "reported_by": "page", **record,
           **({"suppressed_count": suppressed} if suppressed else {})}
    path = request_drive_root(request) / "logs" / "events.jsonl"
    if not await run_sync_to_completion(append_jsonl, path, redact_projection(row).value):
        return json_error("the widget fault row was not written", 503)
    return JSONResponse({"ok": True, "logged": True})


__all__ = [
    "ExtensionLiveSnapshot",
    "WidgetFaultRequest",
    "WidgetFaultResponse",
    "WidgetTab",
    "WidgetsResponse",
    "api_widget_fault_post",
    "api_widgets",
    "widget_tabs",
]
