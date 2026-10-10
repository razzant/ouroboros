"""``POST /api/owner/effort-range``: the owner's effort range from the chat composer.

The one writer of the three range keys (``settings_scales.EFFORT_RANGE_KEYS``): a complete
ordered triple of tiers, written atomically under the owner write seam, projected into the
process environment under the same lock and audited — the contour of the context-mode
endpoint without its idle rule, because the range must be movable while work runs (a
running participant keeps the range it started with; the next one reads this).
"""

from __future__ import annotations

import os
from typing import Any, Dict

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway.owner_settings import (
    _owner_audit,
    _owner_read_settings_raw,
    _owner_update_settings,
    owner_write_guard,
    settings_document_digest,
    settings_document_mutation,
    unsaved_error,
)
from ouroboros.settings_scales import EFFORT_RANGE_KEYS, EFFORT_SCALE, effort_range, effort_rank

# The body's fields, in the order of ``EFFORT_RANGE_KEYS`` (min, recommended = TASK, max).
RANGE_FIELDS = ("min", "recommended", "max")


def effort_range_refusal(body: Any) -> str:
    """'' when ``body`` is a complete ordered triple of ``EFFORT_SCALE`` tiers, else the sentence
    the 400 carries (every tier is accepted, ``minimal`` included; order is min ≤ recommended ≤ max)."""
    if not isinstance(body, dict):
        return "JSON body must be an object with min, recommended and max."
    tiers: Dict[str, str] = {}
    for field in RANGE_FIELDS:
        value = str(body.get(field) or "").strip().lower()
        if value not in EFFORT_SCALE:
            return f"'{field}' must be one of: {', '.join(EFFORT_SCALE)}."
        tiers[field] = value
    if not (effort_rank(tiers["min"]) <= effort_rank(tiers["recommended"]) <= effort_rank(tiers["max"])):
        return "The range must be ordered min ≤ recommended ≤ max."
    return ""


@owner_write_guard
async def api_owner_effort_range(request: Request) -> JSONResponse:
    """Persist the owner's effort range; every participant started afterwards reads it."""
    from ouroboros.gateway.settings import _json_body_or_empty, _run_settings_writer

    body = await _json_body_or_empty(request)
    # Off the event loop, under the document lock (held inside), like every settings writer.
    return await _run_settings_writer(_api_owner_effort_range_sync, request, body)


def _api_owner_effort_range_sync(request: Request, body: Any) -> JSONResponse:
    refusal = effort_range_refusal(body)
    if refusal:
        return unsaved_error(refusal, 400, code="effort_range_invalid")
    values = {key: str(body[field]).strip().lower() for key, field in zip(EFFORT_RANGE_KEYS, RANGE_FIELDS)}

    def _set_range(current: Dict[str, Any]) -> Dict[str, Any]:
        current.update(values)
        return current

    with settings_document_mutation():
        # The previous range is read for the audit trail only; the digest binds that read
        # to the document the locked update replaces.
        digest = settings_document_digest()
        previous = effort_range(_owner_read_settings_raw())
        _owner_update_settings(_set_range, digest, authored_keys=EFFORT_RANGE_KEYS)
        # Same-lock projection: the server process reads the range from its environment.
        for key, value in values.items():
            os.environ[key] = value
    current = effort_range(values)
    _owner_audit(request, "effort_range", {"effort_range": current, "previous_effort_range": previous})
    return JSONResponse({"ok": True, "effort_range": current})


__all__ = ["RANGE_FIELDS", "api_owner_effort_range", "effort_range_refusal"]
