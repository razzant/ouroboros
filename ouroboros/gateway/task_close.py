"""HTTP acknowledgements of exact saved-park closure (no ownership decisions).

Each answer states only what is proven for THIS request: by its own driver's return in this call (which drives this
exact open Close and nothing else: never an ordinary Stop that replaced it, ``drive_request``) or by its own durable
record. 200: this Close closed the park (or a retry reads that recorded Close); 200 ``result_kept``: this request
found the attempt already ended by itself, or released an answered root's late-work pause (the fence names this
request, so a retry reads only that one again), so that result or answer stands. 202: only this exact request is
still owed. 409: this call provably made no change (``expired_park_changed``), or closed some frozen members and left
changed work (``expired_park_partly_closed``); nothing here speaks for an earlier request. 503
``cancel_intent_projection_corrupt``: the cancel-intent projection is corrupt and retrying cannot help until it is
repaired; before this request was recorded nothing was requested, after it (``request_id`` present) its outcome is
unknown. Any other 503: this call's outcome is not confirmed (neither closed nor nothing closed), including a tree its
reading could not open; the same Close may be retried."""

import logging
from typing import Any

from starlette.responses import JSONResponse
from ouroboros.gateway._helpers import json_error

log = logging.getLogger(__name__)
_NO_CHANGE = "The saved work changed. This Close made no change."


def _unconfirmed(task_id: str, request_id: str = "") -> JSONResponse:
    return json_error("The Close outcome is not confirmed yet. Retry the same Close.", 503, task_id=task_id,
                      **({"request_id": request_id} if request_id else {}))


def _corrupt(task_id: str, request_id: str = "") -> JSONResponse:
    """Known corruption, never a retry hint. After the request was recorded its outcome is unknown, not "nothing"."""
    state = ("this Close was recorded, but its outcome cannot be read" if request_id else "nothing was requested")
    return json_error(f"The cancel-intent projection (state/cancel_intents.json) is corrupt; {state}. Retrying cannot "
                      "succeed until the file is repaired; the malformed file was preserved.", 503, task_id=task_id,
                      reason_code="cancel_intent_projection_corrupt", **({"request_id": request_id} if request_id else {}))


def close_expired_response(task_id: str, token: Any) -> JSONResponse:
    from ouroboros.cancel_intents import CancelIntentProjectionCorrupt, active_intent, conditional_park_close
    from supervisor import queue
    from supervisor.expired_park_close import CHANGED, PARTIAL, drive_request, request_close

    try:
        asked = request_close(queue, task_id, token)
    except CancelIntentProjectionCorrupt:
        log.error("Close of %s refused: the cancel-intent projection is corrupt", task_id)
        return _corrupt(task_id)
    except Exception:
        log.warning("Expired park was not confirmed for %s", task_id, exc_info=True)
        return _unconfirmed(task_id)
    if not asked["ok"]:
        if asked["error"] == "invalid_close_expired":
            return json_error("An exact expired park offer is required", 400, task_id=task_id)
        return json_error(_NO_CHANGE, 409, task_id=task_id, reason_code=CHANGED)
    request_id = asked["request_id"]
    own = asked.get("settled") or drive_request(queue, task_id, request_id)  # None: this call drove nothing
    # Outcomes this request's own driver produced need no further read.
    if own == PARTIAL:
        return json_error("Some parked members were closed; changed work was left alone.", 409, task_id=task_id,
                          reason_code=PARTIAL)
    if own in {CHANGED, queue.CANCEL_NOT_FOUND}:
        return json_error(_NO_CHANGE, 409, task_id=task_id, reason_code=own)
    if own not in {queue.CANCEL_CANCELLED, queue.CANCEL_ALREADY_SETTLED}:  # unresolved here: is the debt still ours?
        try:
            pending = active_intent(queue.DRIVE_ROOT, task_id, strict=True, include_close=True)
        except CancelIntentProjectionCorrupt:
            return _corrupt(task_id, request_id)
        except Exception:
            log.debug("Close of %s: its request was not read", task_id, exc_info=True)
            return _unconfirmed(task_id, request_id)
        if conditional_park_close(pending) and pending.get("request_id") == request_id:
            return JSONResponse({"ok": True, "task_id": task_id, "request_id": request_id,
                                 "cancel_state": "pending"}, status_code=202)
    record = _own_record(task_id, token, request_id)
    if record is None:
        return _unconfirmed(task_id, request_id)
    closed, kept, status = record
    if closed:
        return JSONResponse({"ok": True, "task_id": task_id, "request_id": request_id})
    if kept or own == queue.CANCEL_ALREADY_SETTLED and status not in {"", "cancelled"}:
        return JSONResponse({"ok": True, "task_id": task_id, "request_id": request_id, "reason_code": "result_kept"})
    if own == queue.CANCEL_ALREADY_SETTLED:  # this driver found it cancelled by another hand: this Close changed nothing
        return json_error(_NO_CHANGE, 409, task_id=task_id, reason_code=CHANGED)
    return _unconfirmed(task_id, request_id)  # no proof of this request's disposition


def _own_record(task_id: str, token: Any, request_id: str) -> tuple[bool, bool, str] | None:
    """From the task's durable record: whether this exact offer (or request) closed it, whether this request
    released its late-work pause (its answer kept), and its stored status; ``None`` when unread (unknown)."""
    from ouroboros.task_results import load_task_result
    from supervisor import queue
    from supervisor.expired_park_close import _kept_late, _settled_close

    try:
        stored = load_task_result(queue.DRIVE_ROOT, task_id, strict=True) or {}
    except Exception:
        log.debug("Close of %s: retained result unread", task_id, exc_info=True)
        return None
    return (bool(_settled_close(stored, token, request_id)), bool(_kept_late(stored, token, request_id)),
            str(stored.get("status") or ""))
