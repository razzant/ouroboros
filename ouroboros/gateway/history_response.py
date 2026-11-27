"""The canonical history response and its HTTP adapter.

The same assembly returns status, the existing JSON bytes and decoded cursor
kind in either interpreter. Authentication, attribution and gzip stay in ASGI.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros.gateway._helpers import coerce_int, request_profile_scope
from ouroboros.gateway.history_paging import DECODED_CURSOR, HistoryCursorError, RegistryUnavailable, failed_read_facts
from ouroboros.store import StoreError, StoreMigrating


def refusal(reason, cursor, detail=None):
    return JSONResponse({"messages": [], "error": reason, "reason_code": reason, **(detail or {}),
                         "next_cursor": cursor, "has_more": True,
                         "window": {"complete": False, "truncated_by": [reason]}}).body


def assemble(build, data_dir, thread_id, n_human, n_progress, cursor, around, profile_scope, *, live=None):
    decoded = DECODED_CURSOR.get()
    cursor_token = None
    if decoded is None:  # an interpreter without an HTTP observer owns its own decode result
        decoded = {}
        cursor_token = DECODED_CURSOR.set(decoded)
    try:
        try:
            body = build(data_dir, thread_id, n_human, n_progress, cursor, around, profile_scope,
                         **({"live": live} if live is not None else {}))
        except (HistoryCursorError, OSError, StoreError) as exc:
            reason = (exc.reason if isinstance(exc, HistoryCursorError) else
                      "migrating" if isinstance(exc, StoreMigrating) else "history_source_unavailable")
            detail = {"detail": str(exc)} if type(exc) in (StoreError, RegistryUnavailable) else {}
            detail.update(failed_read_facts(exc, data_dir, thread_id, not (cursor or around), profile_scope))
            return (exc.status if isinstance(exc, HistoryCursorError) else 503,
                    refusal(reason, cursor, detail), decoded.get("kind"))
        return 200, body, decoded.get("kind")
    finally:
        if cursor_token is not None:
            DECODED_CURSOR.reset(cursor_token)


def make_history_endpoint(data_dir, build, human_quota, progress_quota):
    from ouroboros.gateway import history_reader

    reader_root = Path(data_dir).resolve()  # bound at route construction, not on the HTTP event loop

    async def api_chat_history(request: Request) -> Response:
        """Return recent chat, system, and progress messages merged chronologically."""
        params = request.query_params

        def _int_param(name: str, default: int, cap: int) -> int:
            return max(0, min(coerce_int(params.get(name), default), cap))

        # Separate quotas: a burst of progress never evicts the real conversation.
        # The legacy limit sets the human default; explicit n_human wins.
        default_human, max_human = human_quota
        n_human = _int_param("n_human", _int_param("limit", 0, max_human) or default_human, max_human)
        n_progress = _int_param("n_progress", *progress_quota)
        # Place conversation ids exceed 32 bits but remain exact on the web client.
        thread_id = _int_param("chat_id", 1, 2**53 - 1) or 1
        selectors = dict(thread_id=thread_id, n_human=n_human, n_progress=n_progress, cursor=params.get("cursor"),
                         around=params.get("around"), profile_scope=request_profile_scope(request))
        inline = partial(assemble, build, data_dir, **selectors)
        status, body, _kind = await history_reader.reader.read(reader_root, selectors, inline,
                                                               receive=getattr(request, "receive", None))
        return Response(content=body, media_type="application/json", status_code=status)

    return api_chat_history
