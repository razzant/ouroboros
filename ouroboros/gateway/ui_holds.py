"""``POST /api/ui/holds``: a page names the foreign UI it held back when it opened.

The start ledger lives in the page (``web/modules/holds.js``): before a skill's page module, a skill's
module or route frame, or an app card Ouroboros wrote gets its first turn, the page writes the item's
key; a key the next page finds means the page did not outlive that start (it froze, crashed, or the
app quit before the start settled). That page holds the item and posts the keys it found, once. The host
names them in one System row (``system_type: "ui_hold"``) in the chat that owns them, the room the card
was published in or the owner's chat for a skill's page code and widgets, so the owner reads it on every
surface and Ouroboros reads it in its next turn's context: a fact, never a reaction. A ``key@rev`` is named
once per server process, so another tab or device that finds the same hold adds no row.
"""

from __future__ import annotations

import logging
import pathlib
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import json_error, request_drive_root, request_json_or, run_sync_to_completion

log = logging.getLogger(__name__)

MAX_REPORTED_HOLDS = 64  # the ledger's own bound (web/modules/holds.js MAX_ENTRIES)
_KEY = re.compile(r"^(app|widget|module):\S{1,280}$")
_NAMED: "OrderedDict[str, None]" = OrderedDict()
_NAMED_MAX = 512
_NAMING: set = set()  # markers a report is naming right now: another tab's report of one waits for its next load


class _ChatUnavailable(RuntimeError):
    """No chat writer serves this data root (no running supervisor): nothing was written."""



def _surface_words(surface: Any) -> str:
    facts = surface if isinstance(surface, dict) else {}
    if facts.get("pywebview") is True:
        return "the desktop app"
    if facts.get("telegram") is True:
        return "the Telegram mini app"
    return "a browser"


def _item(root: pathlib.Path, key: str) -> Tuple[str, Optional[int]]:
    """The words for one held item and the chat that owns it (None: Main)."""
    kind, _, rest = key.partition(":")
    if kind == "app":
        from ouroboros.card_store import load_card

        record = load_card(root, rest)
        if record is None:
            return "", None
        chat_id = record.get("publication_chat_id", record.get("chat_id"))
        return f"the card «{record.get('title') or rest}»", chat_id if isinstance(chat_id, int) else None
    if kind == "widget":
        skill, _, tab = rest.partition(":")
        return f"the {tab} widget of {skill}", None
    return f"the page code of {rest.partition(':')[0]}", None


def _remember(markers: List[str]) -> None:
    for marker in markers:
        _NAMED[marker] = None
    while len(_NAMED) > _NAMED_MAX:
        _NAMED.popitem(last=False)


def _name_holds(root: pathlib.Path, groups: Dict[Optional[int], List[Tuple[str, str]]], surface: str) -> int:
    """One confirmed row per chat, its holds remembered as soon as it is written (a report whose request was
    cancelled meanwhile still counts); how many holds were named. An unconfirmed write raises."""
    from supervisor import message_bus
    from supervisor.schedule_notes import owner_chat_id

    data_root = message_bus.DATA_DIR
    if data_root is None or pathlib.Path(data_root).resolve() != root.resolve() or message_bus.try_get_bridge() is None:
        raise _ChatUnavailable("no chat writer serves this data root")
    named = 0
    for chat_id, items in groups.items():
        names = ", ".join(words for words, _marker in items)
        verb = "was" if len(items) == 1 else "were"
        message_bus.send_with_budget(
            chat_id if chat_id is not None else owner_chat_id(root),
            f"In {surface}, the page stopped while {names} {verb} starting.",
            role="system", system_type="ui_hold", require_write=True, ensure_record_boundary=True,
        )
        _remember([marker for _words, marker in items])
        named += len(items)
    return named


async def api_ui_holds_post(request: Request) -> JSONResponse:
    """``{"holds": [{"key", "rev"}], "surface": {"pywebview", "telegram"}}`` → ``200 {ok, named}``.

    ``400`` a body that is not this shape; ``503`` ``chat_unavailable`` or ``in_flight`` (another report is naming
    one of these now; nothing was named) or ``not_confirmed`` (a row may be missing); after a 503 the next page
    reports again."""
    body = await request_json_or(request, None)
    holds = body.get("holds") if isinstance(body, dict) else None
    if not isinstance(holds, list) or len(holds) > MAX_REPORTED_HOLDS:
        return json_error(f"holds must be a list of at most {MAX_REPORTED_HOLDS} items", 400)
    root = request_drive_root(request)
    groups: Dict[Optional[int], List[Tuple[str, str]]] = {}
    fresh: List[str] = []
    for hold in holds:
        key = hold.get("key") if isinstance(hold, dict) else None
        if not isinstance(key, str) or not _KEY.match(key):
            return json_error("each hold needs a key: app:, widget: or module:", 400)
        marker = f"{key}@{str(hold.get('rev') or '')[:128]}"
        if marker in _NAMING:  # unwritten yet: answering 200 here would let that tab mark it reported
            return json_error("another report is naming this hold right now", 503, reason_code="in_flight")
        if marker in _NAMED or marker in fresh:
            continue
        words, chat_id = _item(root, key)
        if words:
            groups.setdefault(chat_id, []).append((words, marker))
            fresh.append(marker)
    if not groups:
        return JSONResponse({"ok": True, "named": 0})
    _NAMING.update(fresh)
    try:
        named = await run_sync_to_completion(_name_holds, root, groups, _surface_words(body.get("surface")))
    except _ChatUnavailable as exc:
        return json_error(str(exc), 503, reason_code="chat_unavailable")
    except Exception:  # the rows before it are remembered; the next page reports the rest again
        log.warning("A ui_hold row was not confirmed", exc_info=True)
        return json_error("a hold row was not confirmed", 503, reason_code="not_confirmed")
    finally:
        _NAMING.difference_update(fresh)
    return JSONResponse({"ok": True, "named": named})


__all__ = ["MAX_REPORTED_HOLDS", "api_ui_holds_post"]
