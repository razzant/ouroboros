"""The local Host Service return door of a reviewed connector's own outgoing call."""
from __future__ import annotations

import functools
import json

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.extension_late_result import (
    LateResultUnavailable, LateResultWaiting, binding_for_report, check_address, inline_delivery, project_handoff,
    report_input,
)
from ouroboros.store import StoreError


def _accept(ctx, skill: str, content_hash: str, report_id: str, fingerprint: str, text: str,
            author: dict, letter_id: str) -> str:
    from supervisor.message_ingress import LATE_RESULT_DISPATCHER, accept_local_message, dispatch_facts
    from ouroboros.process_custody import current_custody_session_id

    binding = binding_for_report(ctx.data_dir, report_id)
    if (binding.get("skill"), binding.get("content_hash")) != (skill, content_hash):
        raise LateResultUnavailable("call not recorded")
    if delivery := inline_delivery(ctx.data_dir, binding, fingerprint):
        return delivery
    chat_id = binding["chat_id"]
    if decision := binding.get("handoff"):
        project_handoff(ctx.data_dir, chat_id, report_id, decision)
    row, _ = accept_local_message(
        ctx.bridge_getter(), ctx.data_dir, text, author=author, report_digest=fingerprint,
        letter_id=letter_id, chat_id=chat_id, user_id=0, source="web", client_message_id=report_id,
        before_accept=lambda: check_address(ctx.data_dir, binding), room_echo=bool(binding["project_id"]),
    )
    facts = dispatch_facts(row, ctx.data_dir) or {}
    if facts.get("dispatcher") != LATE_RESULT_DISPATCHER:
        raise LateResultUnavailable("consideration not confirmed")  # an earlier body confirmed only its queue
    outcome = facts.get("delivery") if facts.get("state") == "completed" else None
    if outcome == "handed_off":
        return outcome
    if facts.get("state") == "pending" or (facts.get("state") == "dispatched"
                                             and facts.get("process") == current_custody_session_id()):
        raise LateResultWaiting("consideration is still owed")
    raise LateResultUnavailable("consideration not confirmed")


async def api_late_result(request: Request) -> JSONResponse:
    from ouroboros.gateway import host_service as host
    from supervisor.message_ingress import ClientMessageIdConflict

    ctx = request.app.state.host_service_context
    try:
        skill, token = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError:
        return JSONResponse({"ok": False, "code": "not_available"}, status_code=403)
    except OSError:
        return _retry()
    lane = f"{skill}:late_result"
    if not ctx.rate_limiter.allow(lane) or not ctx._enter_inflight(lane):
        return _retry()
    try:
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 128 * 1024:
                return JSONResponse({"ok": False, "code": "invalid_report"}, status_code=413)
        report = report_input(json.loads(raw))
        delivery = await host._bounded_host_read(ctx, functools.partial(
            _accept, ctx, skill, str(token.get("content_hash") or ""), *report,
        ))
    except ClientMessageIdConflict:
        return JSONResponse({"ok": False, "code": "report_conflict"}, status_code=409)
    except LateResultUnavailable:
        return JSONResponse({"ok": False, "code": "not_available"}, status_code=409)
    except (LateResultWaiting, OSError, StoreError, RuntimeError):
        return _retry()
    except (ValueError, UnicodeError):
        return JSONResponse({"ok": False, "code": "invalid_report"}, status_code=400)
    finally:
        ctx._leave_inflight(lane)
    return JSONResponse({"ok": True, "report_ref": report[0], "delivery": delivery})


def _retry() -> JSONResponse:
    return JSONResponse({"ok": False, "code": "retry", "retry_after_s": 30}, status_code=429)
