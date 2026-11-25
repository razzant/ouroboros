"""Why each ``GET /api/chat/history`` was made: one ``history_read`` line in ``server.log`` per request.

A client names on a read the action that began it (``reason``), its cursor's kind (``mode``), the gate or
fallback leg that chose it (``decision``) and what the earlier leg met (``detail``); the line adds the room, the
status, the refusal the server gave (``code``, ``refusal``, ``stream``), the answer's bytes and its milliseconds
in the endpoint (CLIENT_CONTRACT "Reading history"). Telemetry only: nothing selects a page, authorizes or checks
a cursor by these tags, and the answer is the same with or without them. A tag outside its closed set is recorded
``invalid``, an absent reason or cursor kind ``unspecified``; no tag is echoed, and no cursor, text, address or
token is written. The line rides the server's bounded stdlib log (PERSISTENCE ``logs/server.log``)."""

from __future__ import annotations

import functools
import json
import logging
import re
import time

from ouroboros.gateway._helpers import coerce_int
from ouroboros.gateway.history_paging import DELTA_RESET_DETAILS

log = logging.getLogger(__name__)

REASONS = frozenset({"open", "settled", "bound_recheck", "resume", "reconnect", "read_receipt", "navigate",
                     "refresh", "retry"})
DECISIONS = frozenset({"explicit_whole", "latest_unknown", "not_loaded", "prior_failed", "catch_up_owed",
                       "live_card_bound", "no_cursor", "window_overflow", "horizon_gap", "view_changed",
                       "archive_return", "base_superseded", "after_refused", "after_failed", "after_rejected"})
# What an earlier leg met: the server's refusal (a changed view; an older server's reset without a cause), a 200 the
# client could not admit or apply, a failed read (also ``http_<status>``) or a store still importing.
DETAILS = DELTA_RESET_DETAILS | {"view_changed", "delta_reset", "bad_messages", "no_continuation", "stream_gap",
                                 "latest_unknown", "profile_mismatch", "apply_failed", "network", "migrating"}
_CURSOR_KINDS = frozenset({"after", "older", "page"})
_HTTP_STATUS = re.compile(r"http_[1-5][0-9]{2}")
_SERVER_WORD = re.compile(r"[a-z_]{1,48}")  # the server's own codes, causes and stream names


def _tag(value, allowed) -> str:
    return "unspecified" if not value else value if value in allowed else "invalid"


def _fields(params, response, seconds: float) -> dict:
    around, cursor = params.get("around"), params.get("cursor")
    fields = {"room": max(0, min(coerce_int(params.get("chat_id"), 1), 2**53 - 1)) or 1,
              "reason": _tag(params.get("reason"), REASONS),
              # The newest page and a page around one place are the request's own; a cursor is opaque here.
              "mode": ("invalid" if cursor else "around") if around is not None
              else _tag(params.get("mode"), _CURSOR_KINDS) if cursor else "newest"}
    for name, allowed in (("decision", DECISIONS), ("detail", DETAILS)):
        if value := params.get(name):  # where no gate or fallback chose the read, the client names none
            fields[name] = value if value in allowed or (name == "detail" and _HTTP_STATUS.fullmatch(value)) \
                else "invalid"
    status = getattr(response, "status_code", None)
    fields["status"] = status or "none"  # none: the endpoint answered nothing (an exception, a cancelled read)
    if status not in (None, 200):
        answer = json.loads(response.body)
        named = {"code": answer.get("reason_code"), **({"refusal": answer.get("detail"),
                                                         "stream": answer.get("stream")} if status == 409 else {})}
        fields.update({key: value for key, value in named.items() if _SERVER_WORD.fullmatch(str(value or ""))})
    fields["bytes"] = len(response.body) if response is not None else 0  # uncompressed: before gzip and delivery
    fields["ms"] = round(seconds * 1000)
    return fields


def recorded(endpoint):
    """The history route's ``endpoint`` with each request it serves recorded; a failed record never fails the read."""
    @functools.wraps(endpoint)
    async def read(request):
        started, response = time.perf_counter(), None
        try:
            response = await endpoint(request)
            return response
        finally:
            try:
                fields = _fields(request.query_params, response, time.perf_counter() - started)
                log.info("history_read %s", " ".join(f"{key}={value}" for key, value in fields.items()))
            except Exception:
                log.debug("A history read's record failed", exc_info=True)
    return read
