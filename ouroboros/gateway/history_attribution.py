"""Why each ``GET /api/chat/history`` was made and what it cost, counted in memory: the ``history_reads`` section of
the next ``supervisor_load`` row (``server_liveness._load_sections``; PERSISTENCE ``logs/supervisor.jsonl``).

A client names on a read the action that began it (``reason``), its cursor's kind (``mode``), the gate or
fallback leg that chose it (``decision``) and what the earlier leg met (``detail``) (CLIENT_CONTRACT "Reading
history"). Telemetry only: nothing selects a page, authorizes or checks a cursor by these tags, the answer is the
same with or without them, and a read writes nothing (DEVELOPMENT "Passive GET"). A read's cause is its counted
words: each tag as a value of its closed set (one outside it is ``invalid``, an absent reason or cursor kind
``unspecified``; a cursor's kind is the one the selection decoded, ``history_paging.DECODED_CURSOR``, where it
decoded one), its status and the refusal the server gave (``code``, ``refusal``, ``stream``). A window keeps one
group per distinct cause with its reads, the answers' bytes and their milliseconds in the endpoint (total,
longest), so each cost stays with what caused it; over all its reads, the same totals and a fixed histogram for
their median and tail. Never a room, a tag as sent, a cursor, text, address or token. Not counted: a HEAD, and a
request answered before the route (an expired session's 401, a refused origin's 403)."""

from __future__ import annotations

import functools
import json
import logging
import re
import threading
import time

from ouroboros.gateway.history_paging import DECODED_CURSOR, DELTA_RESET_DETAILS

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
# The time histogram's buckets, each an upper bound in milliseconds (the last open): a measurement's fixed scale, so
# a window's median and tail read off its counts; never a gate.
BUCKETS = (("<0.5", 500), ("<1", 1000), ("<2", 2000), ("<5", 5000), ("<10", 10_000), (">=10", None))
# A window's distinct causes beyond this many share one ``overflow`` group, their reads and cost kept: a bound on the
# row's size in transport (a busy minute names a handful), never on what is counted.
GROUPS_MAX = 64
_OVERFLOW = (("overflow", True),)  # never a cause: every cause names its reason, mode and status
_LOCK = threading.Lock()  # the route counts on the event loop, the supervisor loop drains on its own thread
_WINDOW: dict = {}


def _tag(value, allowed) -> str:
    return "unspecified" if not value else value if value in allowed else "invalid"


def _fields(params, response, kind=None) -> dict:
    """One read's counted words: its tags as values of their sets, its status and the refusal the server named."""
    around, cursor = params.get("around"), params.get("cursor")
    fields = {"reason": _tag(params.get("reason"), REASONS),
              # The newest page and a page around one place are the request's own; a cursor's kind is the one the
              # selection decoded (``kind``), the client's word only where it decoded none.
              "mode": ("invalid" if cursor else "around") if around is not None
              else (kind or _tag(params.get("mode"), _CURSOR_KINDS)) if cursor else "newest"}
    for name, allowed in (("decision", DECISIONS), ("detail", DETAILS)):
        if value := params.get(name):  # where no gate or fallback chose the read, the client names none
            fields[name] = value if value in allowed or (name == "detail" and _HTTP_STATUS.fullmatch(value)) \
                else "invalid"
    status = getattr(response, "status_code", None)
    fields["status"] = str(status or "none")  # none: the endpoint answered nothing (an exception, a cancelled read)
    if status not in (None, 200):
        try:
            answer = json.loads(response.body)
        except (AttributeError, TypeError, ValueError):  # a body that is not the error page: the read still counts
            answer = None
        if isinstance(answer, dict):
            named = {"code": answer.get("reason_code"), **({"refusal": answer.get("detail"),
                                                             "stream": answer.get("stream")} if status == 409 else {})}
            fields.update({key: value for key, value in named.items() if _SERVER_WORD.fullmatch(str(value or ""))})
    return fields


def _count(fields: dict, size: int, ms: int) -> None:
    cause = tuple(sorted(fields.items()))
    with _LOCK:
        if not _WINDOW:
            _WINDOW.update(n=0, bytes=0, total_ms=0, max_ms=0, seconds={label: 0 for label, _bound in BUCKETS},
                           groups={})
        groups = _WINDOW["groups"]
        if cause not in groups and len(groups) - (_OVERFLOW in groups) >= GROUPS_MAX:
            cause, fields = _OVERFLOW, dict(_OVERFLOW)
        group = groups.setdefault(cause, {**fields, "n": 0, "bytes": 0, "total_ms": 0, "max_ms": 0})
        for totals in (group, _WINDOW):
            totals["n"] += 1
            totals["bytes"] += size  # uncompressed: before gzip and delivery
            totals["total_ms"] += ms
            totals["max_ms"] = max(totals["max_ms"], ms)
        _WINDOW["seconds"][next(label for label, bound in BUCKETS if bound is None or ms < bound)] += 1


def drain() -> dict:
    """The history reads counted since the last drain (``{}`` for none), then an empty window: its totals, the time
    histogram and ``groups``, one per cause in the order first counted (``overflow`` last when a window had one)."""
    with _LOCK:
        window = dict(_WINDOW)
        _WINDOW.clear()  # the next read starts new totals, histogram and groups: the drained ones are this copy's
    if window:
        window["groups"] = list(window["groups"].values())
    return window


def recorded(endpoint):
    """The history route's ``endpoint`` with each GET it serves counted; counting never fails or delays the read."""
    @functools.wraps(endpoint)
    async def read(request):
        started, response, decoded = time.perf_counter(), None, {}
        token = DECODED_CURSOR.set(decoded)  # the read's thread shares this mapping (``asyncio.to_thread`` context)
        try:
            response = await endpoint(request)
            return response
        finally:
            DECODED_CURSOR.reset(token)
            if getattr(request, "method", "GET") == "GET":  # a HEAD's answer carries no page
                try:
                    _count(_fields(request.query_params, response, decoded.get("kind")),
                           len(response.body) if response is not None else 0, round((time.perf_counter() - started) * 1000))
                except Exception:
                    log.debug("A history read was not counted", exc_info=True)
    return read
