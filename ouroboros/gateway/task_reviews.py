"""Authenticated transport for task-owned review history and exact saved text.

The ledger, plan record and skill-history owners resolve selections and custody;
this gateway validates query syntax and exports complete safe text. All routes
inherit NetworkAuthGate. Archived history is read only on explicit requests.
"""
from __future__ import annotations

import hashlib

from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ouroboros import review_ledger as ledger
from ouroboros.gateway._helpers import json_error, request_drive_root, run_sync_to_completion
from ouroboros.observability import redact_projection
from ouroboros.review_projection import task_review_history
from ouroboros.task_results import validate_task_id


def _history(request):
    try:
        task_id = validate_task_id(request.path_params["task_id"])
        offset = int(request.query_params.get("offset", "0"))
        limit = int(request.query_params.get("limit", "20"))
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("offset must be nonnegative; limit must be 1–50")
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        payload = task_review_history(request_drive_root(request), task_id, offset=offset, limit=limit)
    except Exception:
        return json_error("Review history could not be read.", 503, reason_code="review_history_unavailable")
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def _text_export(text, *, sha=""):
    public = str(redact_projection(text).value).encode("utf-8")
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
               "Content-Disposition": 'attachment; filename="review.txt"',
               "X-Review-Export-SHA256": hashlib.sha256(public).hexdigest()}
    if sha:
        headers["X-Review-Source-SHA256"] = sha  # Redacted bytes may have a different digest.
    return Response(public, media_type="text/plain", headers=headers)


def _source(request):
    from ouroboros.plan_record import read_reply_source

    try:
        task_id = validate_task_id(request.path_params["task_id"])
    except ValueError as exc:
        return json_error(str(exc), 400)
    domain = request.query_params.get("domain", "ledger")
    role = request.query_params.get("role", "response")
    part, sha = request.query_params.get("part", ""), request.query_params.get("sha256", "")
    readers = {"ledger": ledger.read_selected_source, "plan": read_reply_source}
    if domain not in readers or role not in {"response", "failure"} or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        return json_error("An exact saved review source is required.", 400, reason_code="review_source_invalid")
    try:
        data = readers[domain](request_drive_root(request), task_id, request.path_params["record_id"],
                               request.path_params["seat_id"], role=role, part=part, sha256=sha)
        return _text_export(data.decode("utf-8"), sha=sha)
    except ledger.ReviewSourceChanged as exc:
        return json_error(str(exc), 409, reason_code="review_source_changed")
    except (OSError, ValueError, TypeError, LookupError):
        return json_error("The saved review text failed its integrity check or cannot be read.", 404,
                          reason_code="review_source_unavailable")


def _skill_source(request):
    from ouroboros.skill_review_history import review_history_detail

    try:
        task_id = validate_task_id(request.path_params["task_id"])
    except ValueError as exc:
        return json_error(str(exc), 400)
    try:
        payload = review_history_detail(request_drive_root(request), request.query_params.get("skill", ""),
                                        request.path_params["job_id"], task_id=task_id)
    except (OSError, ValueError, TypeError):
        return json_error("Saved skill review is unavailable.", 503)
    if payload.get("error"):
        return json_error(payload["error"], payload["status_code"])
    return _text_export(payload["markdown"])


async def api_task_reviews(request):
    return await run_sync_to_completion(_history, request)


async def api_task_review_source(request):
    return await run_sync_to_completion(_source, request)


async def api_task_skill_review_source(request):
    return await run_sync_to_completion(_skill_source, request)


def task_review_routes():
    return [Route("/api/tasks/{task_id}/reviews", api_task_reviews, methods=["GET"]),
            Route("/api/tasks/{task_id}/reviews/skill/{job_id}", api_task_skill_review_source, methods=["GET"]),
            Route("/api/tasks/{task_id}/reviews/{record_id}/{seat_id}", api_task_review_source, methods=["GET"])]
