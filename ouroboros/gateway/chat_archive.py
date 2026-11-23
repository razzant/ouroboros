"""``GET /api/chat/archive``: the read-only archive of saved rooms while Main's history refuses (``room_archive``).

``room=main`` (the default) or a recognized room's chat id, ``before=`` the cursor a page gave. A read
writes nothing and acknowledges nothing. Errors keep one body, ``{error, reason_code}``: 400
``archive_cursor_invalid`` (a cursor this archive never gave for that room), 404 ``archive_room_unknown`` (a
first read of a room no evidence recognizes), 409 ``archive_cursor_changed`` (a cursor it gave, whose room's
selecting evidence changed or whose room is no longer recognized: read that room's newest page again), 503
``migrating`` (the dialogue store or older rows are still being prepared) or ``archive_source_unavailable`` (the
dialogue store or the task lineage cannot answer now).
"""
from __future__ import annotations

import asyncio
import logging
import pathlib

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.room_archive import ArchiveCursorChanged, ArchiveCursorError, ArchiveRoomUnknown, archive_page
from ouroboros.store import StoreError, StoreMigrating

log = logging.getLogger(__name__)


def make_chat_archive_endpoint(data_dir: pathlib.Path):
    async def api_chat_archive(request: Request) -> JSONResponse:
        params = request.query_params
        try:
            body = await asyncio.to_thread(archive_page, data_dir, str(params.get("room") or "main"), params.get("before"))
        except ArchiveCursorError:
            status, reason = 400, "archive_cursor_invalid"
        except ArchiveCursorChanged:
            status, reason = 409, "archive_cursor_changed"
        except ArchiveRoomUnknown:
            status, reason = 404, "archive_room_unknown"
        except StoreMigrating:
            status, reason = 503, "migrating"
        except (OSError, StoreError, ValueError):
            log.debug("Saved-room archive read failed", exc_info=True)
            status, reason = 503, "archive_source_unavailable"
        else:
            return JSONResponse(body, headers={"Cache-Control": "no-store"})
        return JSONResponse({"error": reason, "reason_code": reason}, status_code=status,
                            headers={"Cache-Control": "no-store"})

    return api_chat_archive


__all__ = ["make_chat_archive_endpoint"]
