"""Connector-facing place intake, receipts and correlated work reads."""
from __future__ import annotations

import asyncio
import functools
import json
import logging
from dataclasses import replace
from typing import Any, Dict, TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse
from ouroboros import store
from ouroboros.gateway._helpers import run_sync_to_completion
from ouroboros.presence_delivery import PresenceDeliveryConflict, delivery_reporting_version

if TYPE_CHECKING:
    from ouroboros.gateway import host_service as host

log = logging.getLogger(__name__)

async def _api_presence_delivery(request: Request) -> JSONResponse:
    """Record exact provider receipts without sending or starting model work."""
    from ouroboros.gateway import host_service as host
    ctx: host.HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except OSError:
        return _presence_error("presence authentication unavailable", 500, "presence_auth_unavailable", "retry")
    if not ctx.rate_limiter.allow(f"{skill_name}:presence_delivery"):
        return _presence_error("rate limit exceeded", 429, "presence_rate_limited", "retry")
    # Receipts, turns and inject have separate in-flight budgets: five long turns must
    # not starve the receipts those turns' own sends produce.
    if not ctx._enter_inflight(f"{skill_name}:delivery"):
        return _presence_error("too many in-flight presence requests", 429, "presence_capacity_full", "retry")
    try:
        payload = await request.json()
        result = await run_sync_to_completion(ctx.presence_deliveries.record, skill_name, payload)
        return JSONResponse(result)
    except PresenceDeliveryConflict as exc:
        return _presence_error(str(exc), 409, "presence_receipt_conflict", "rejected")
    except (ValueError, TypeError) as exc:
        return _presence_error(str(exc), 400, "presence_receipt_invalid", "rejected")
    except store.StoreMigrating:  # the receipt's earlier rows are in saved history that is still being upgraded
        return _presence_error("presence delivery history is still importing; retry the same receipt", 503,
                               "presence_history_importing", "retry")
    except Exception:
        log.warning("Presence delivery history write failed for skill %s", skill_name, exc_info=True)
        return _presence_error("presence delivery history write failed; retry the same receipt", 503,
                               "presence_receipt_unwritable", "retry")
    finally:
        ctx._leave_inflight(f"{skill_name}:delivery")


def _presence_error(message: str, status: int, code: str, disposition: str, **facts: Any) -> JSONResponse:
    """Add transport recovery facts without changing the legacy HTTP/error contract.

    The producer chooses semantics, not the status class: a stale token is blocked,
    an origin mismatch rejected, and a failed history write retryable.
    """
    return JSONResponse({"ok": False, "error": message, "code": code,
                         "disposition": disposition, **facts}, status_code=status)


def _presence_exception(exc: Exception, status: int) -> JSONResponse:
    code = str(getattr(exc, "code", "presence_admission_failed"))
    disposition = "blocked"
    if code in {"chat_log_unwritable", "presence_result_missing", "presence_bindings_unreadable",
                "presence_resources_unavailable", "presence_attempt_outcome_unknown",
                "presence_result_unreadable", "presence_event_identity_unproven",
                "presence_start_unwritable"}:
        disposition = "retry"
    elif code in {"presence_attachment_admission_rejected", "presence_binding_wrong_transport",
                  "presence_conversation_key_required", "presence_admission_conversation_mismatch",
                  "presence_event_identity_conflict", "presence_origin_mismatch"}:
        disposition = "rejected"
    if getattr(exc, "disposition", "") in {"retained", "retry"}:
        disposition = exc.disposition  # the runner holds the durable attempt authority
    facts = {"field": getattr(exc, "field", "presence")}
    if getattr(exc, "turn_ref", ""):
        facts["turn_ref"] = exc.turn_ref
        facts["delivery_reporting_version"] = getattr(exc, "delivery_reporting_version", 0)
    if getattr(exc, "retry_after", None) is not None:
        facts["retry_after"] = exc.retry_after
    if getattr(exc, "work_ref", ""):
        facts["work_ref"] = exc.work_ref
    manifest = getattr(exc, "attachment_manifest", None)
    if isinstance(manifest, list):
        facts["attachment_manifest"] = [dict(row) for row in manifest if isinstance(row, dict)]
    return _presence_error(str(exc), status, code, disposition, **facts)


def _admit_presence(ctx: host.HostServiceContext, skill_name: str, binding_id: str, **origin: str) -> Any:
    from ouroboros.presence_admission import admit_presence_turn

    return admit_presence_turn(
        drive_root=ctx.data_dir,
        authenticated_transport_skill=skill_name,
        binding_id=binding_id,
        **origin,
    )


def _outside_origin(admission: Any, provider: str, account_id: str, conversation_id: str, thread_id: str) -> bool:
    """Whether reported provider facts lie outside the owner-created place/binding origin."""
    origin = admission.origin
    return (provider != origin.transport or account_id != origin.account_id
            or (origin.conversation_id != "*" and conversation_id != origin.conversation_id)
            or bool(origin.thread_id and thread_id != origin.thread_id))


async def _api_presence_turn(request: Request) -> JSONResponse:
    """Capture a connector event and run its matched focus under consciousness authority.

    Authentication, admission and file confinement run off the event loop. The turn is
    host work this request only waits on (``presence_runner.PresenceTurnExecutions``): it
    queues on the gate as a coroutine and runs on its own thread. Every version releases
    the request reservation at a qualified park (or terminal settlement); version 1
    returns the initial envelope while version 0's HTTP waiter stays open until the
    terminal result. Version 2 takes that envelope as soon as its author accepts intake,
    and never queues: a busy conversation or slot is a typed retry. A retry of the same
    event joins the retained execution.
    """

    from ouroboros.gateway import host_service as host
    ctx: host.HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except OSError:
        return _presence_error("presence authentication unavailable", 500, "presence_auth_unavailable", "retry")
    if await host._bounded_host_read(ctx, functools.partial(store.migrating, ctx.data_dir)):  # P1: the turn writes its result
        return _presence_error("the task store is importing; retry shortly", 503, "migrating", "retry")
    from ouroboros.presence_admission import PresenceAdmissionError
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_continuation import continuation_version, turn_response
    from ouroboros.presence_runner import (
        PresenceTurnError,
        PresenceTurnEvent,
        PresenceTurnNotStarted,
        presence_turn_replay,
        presence_turn_task_id,
        presence_event_identity,
    )

    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) - {
            "binding_id", "event", "staged_files", "delivery_reporting_version", "continuation_version",
        }:
            return _presence_error("invalid presence payload", 400, "presence_payload_invalid", "rejected")
        reporting_version = delivery_reporting_version(payload.get("delivery_reporting_version", 0))
        continuing = continuation_version(payload.get("continuation_version", 0))
        event_payload = payload.get("event")
        expected = {
            "source_event_id", "provider", "account_id", "conversation_id", "thread_id",
            "conversation_key", "actor", "conversation", "message", "text",
        }
        if not isinstance(event_payload, dict) or set(event_payload) - {"facts", "identity"} != expected:
            return _presence_error("invalid presence event", 400, "presence_event_invalid", "rejected")

        from ouroboros.places import validate_presence_identity
        try:
            identity = validate_presence_identity(event_payload["identity"]) if "identity" in event_payload else None
        except ValueError as exc:
            return _presence_error(str(exc), 400, "presence_identity_invalid", "rejected")
        actor = event_payload["actor"]
        if not isinstance(actor, dict) or ("authenticated" in actor and type(actor["authenticated"]) is not bool):
            return _presence_error("actor must be an object; authenticated must be a boolean", 400,
                                   "presence_actor_invalid", "rejected")

        provider = str(event_payload.get("provider") or "").strip()
        account_id = str(event_payload.get("account_id") or "").strip()
        conversation_id = str(event_payload.get("conversation_id") or "").strip()
        thread_id = str(event_payload.get("thread_id") or "").strip()
        from ouroboros.places import WAKE_FACTS, declared_wake_signals
        facts = event_payload.get("facts")
        allowed_facts = set(WAKE_FACTS) | declared_wake_signals(skill_name)
        if "facts" in event_payload and (not isinstance(facts, list) or any(
                not isinstance(f, str) or f not in allowed_facts for f in facts)):
            raise ValueError("event.facts must be a list of declared wake facts")
        from ouroboros.presence_history import source_order
        source_order(event_payload["message"])  # the provider's order, as history carries it: invalid is a 400
        admission = await host._bounded_host_read(ctx, functools.partial(
            host._admit_presence, ctx, skill_name, str(payload.get("binding_id") or ""),
            provider=provider, account_id=account_id, conversation_id=conversation_id, thread_id=thread_id))
        if _outside_origin(admission, provider, account_id, conversation_id, thread_id):
            return _presence_error("presence event does not match its owner-created binding", 403,
                                   "presence_origin_mismatch", "rejected")
        event = PresenceTurnEvent(
            source_event_id=str(event_payload["source_event_id"] or "").strip(),
            provider=provider,
            account_id=account_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            # The transport authenticates provider facts, but it does not get
            # to choose the concurrency/history identity. Derive that identity
            # from the binding-checked origin facts above.
            conversation_key=conversation_key(provider, account_id, conversation_id, thread_id),
            actor=dict(actor),
            conversation=(
                dict(event_payload["conversation"])
                if isinstance(event_payload["conversation"], dict)
                else {}
            ),
            message=dict(event_payload["message"]) if isinstance(event_payload["message"], dict) else {},
            text=str(event_payload["text"] or ""),
            delivery_reporting_version=reporting_version,
            continuation_version=continuing,
            facts=tuple(facts) if facts is not None else None,
            identity=identity,
        )
        if not event.source_event_id or not event.conversation_key or not event.actor:
            return _presence_error("presence event is missing identity facts", 400,
                                   "presence_identity_missing", "rejected")
        if "transport_queue" in event.conversation:
            from ouroboros.presence_observations import validate_transport_queue

            validate_transport_queue(event.conversation["transport_queue"], event.conversation_key,
                                     event.source_event_id)
        staged_files = await host._bounded_host_read(
            ctx, functools.partial(host._presence_staged_files, ctx, skill_name, payload.get("staged_files")))
        turn_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
        identity = presence_event_identity(admission.binding_id, event)
        from ouroboros.presence_runner import capture_presence_event
        # Hearing precedes allowance, wake and capacity; cached custody remains the replay authority.
        captured = await host._bounded_host_read(ctx, functools.partial(
            capture_presence_event, admission, event, drive_root=ctx.data_dir, staged_files=staged_files))
        # A settled turn answers from its durable row without queueing behind its conversation.
        result = await host._bounded_host_read(ctx, functools.partial(
            presence_turn_replay, ctx.data_dir, turn_id, event.conversation_key, identity, continuing))
        from ouroboros.presence_runner import _stored_turn
        from ouroboros.conversation_lease import live_author
        # A retry of work already admitted belongs to its frozen custody, even if the
        # owner has since switched Off. Never certify a running/unknown attempt silent.
        started = turn_id in ctx.presence_turns.live() or bool(await host._bounded_host_read(
            ctx, functools.partial(_stored_turn, ctx.data_dir, turn_id, identity)))
        if not started:
            started = await host._bounded_host_read(ctx, functools.partial(
                live_author, event.conversation_key, drive_root=ctx.data_dir)) == turn_id
        if result is None and admission.place is not None and not started:
            from ouroboros.place_intake import author_hears, decide_attention, offer_to_author, settle_heard
            decision, allowance = await host._bounded_host_read(ctx, functools.partial(
                decide_attention, ctx.data_dir, admission, event))
            # The conversation's current author hears it first, wake or background, before any
            # rate, reservation, slot or allowance; its final drain settles what it accepted.
            wake = author_hears(decision, allowance, event)
            if wake is not None and await host._bounded_host_read(ctx, functools.partial(
                    offer_to_author, ctx.data_dir, captured, event, wake=wake)):
                result = await host._bounded_host_read(ctx, functools.partial(settle_heard, ctx.data_dir, captured, event, "current_author"))
            elif decision == "silent":
                result = await host._bounded_host_read(ctx, functools.partial(settle_heard, ctx.data_dir, captured, event, allowance["reason"]))
            elif decision == "retry":
                return _presence_error("Consciousness allowance is unavailable; retry this event.", 429,
                    "consciousness_allowance_exhausted", "retry", retry_after=allowance["retry_after"])
            else:
                admission = replace(admission, allowance=allowance)
        if result is None:
            if not ctx.rate_limiter.allow(f"{skill_name}:presence"):
                return _presence_error("rate limit exceeded", 429, "presence_rate_limited", "retry")
            budget = f"{skill_name}:presence"
            execution, _started = ctx.presence_turns.start_or_join(
                turn_id, identity=identity,
                reserve=functools.partial(ctx._enter_inflight, budget),
                release=functools.partial(ctx._leave_inflight, budget),
                # Version 2 never waits in the gate: a busy resource is a typed retry (``admit_configured_gate``).
                admit=functools.partial(ctx.admit_presence_gate, event.conversation_key, wait=continuing < 2),
                run=lambda lease: ctx.presence_runner(
                    admission=admission, event=event, staged_files=staged_files, admitted=lease,
                ),
            )
            if execution is None:
                return _presence_error("too many in-flight presence requests", 429, "presence_capacity_full", "retry")
            # Versions 1 and 2 take a continuing author's initial envelope; v0 waits for its terminal.
            result = await asyncio.wrap_future(execution.initial if continuing else execution.result)
        return JSONResponse(await host._bounded_host_read(ctx, functools.partial(
            turn_response, result, continuing, ctx.data_dir)))
    except json.JSONDecodeError:
        return _presence_error("invalid json", 400, "presence_payload_invalid", "rejected")
    except (PresenceAdmissionError, PresenceTurnError) as exc:
        return _presence_exception(exc, 403 if exc.code == "presence_origin_mismatch"
                                   else 429 if exc.code == "presence_capacity_full" else 409)
    except PresenceTurnNotStarted as exc:
        return _presence_error(str(exc), 503, "presence_turn_not_started", "retry")
    except OSError as exc:
        return _presence_error(str(exc), 400, "presence_io_unavailable", "retry")
    except ValueError as exc:
        return _presence_error(str(exc), 400, "presence_payload_invalid", "rejected")
    except Exception as exc:
        log.debug("Host service presence turn failed", exc_info=True)
        return _presence_error(str(exc), 500, "presence_turn_unavailable", "retry")


async def _api_presence_history(request: Request) -> JSONResponse:
    """Record a connector's earlier messages as room history (``presence_history``).

    Same token, grant and place admission as a turn, with its own window and in-flight
    budget. No attention, task, wake, mailbox or outbox follows; there is no fallback to
    ``/presence/turn``.
    """
    from ouroboros.gateway import host_service as host
    from ouroboros.presence_admission import PresenceAdmissionError
    from ouroboros.presence_history import PresenceHistoryInvalid, parse_history_request, record_history
    ctx: host.HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except OSError:
        return _presence_error("presence authentication unavailable", 500, "presence_auth_unavailable", "retry")
    if not ctx.rate_limiter.allow(f"{skill_name}:presence_history"):
        return _presence_error("rate limit exceeded", 429, "presence_rate_limited", "retry")
    if not ctx._enter_inflight(f"{skill_name}:history"):
        return _presence_error("too many in-flight presence requests", 429, "presence_capacity_full", "retry")
    try:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):  # never a retryable outage
            raise PresenceHistoryInvalid("payload", "the body must be JSON text in UTF-8") from None
        parsed = parse_history_request(payload)
        first = parsed.records[0].event
        admission = await host._bounded_host_read(ctx, functools.partial(
            host._admit_presence, ctx, skill_name, parsed.binding_id, provider=first.provider,
            account_id=first.account_id, conversation_id=first.conversation_id, thread_id=first.thread_id))
        if _outside_origin(admission, first.provider, first.account_id, first.conversation_id, first.thread_id):
            return _presence_error("presence history does not match its owner-created binding", 403,
                                   "presence_origin_mismatch", "rejected")
        status, body = await run_sync_to_completion(record_history, ctx.data_dir, admission, parsed)
        return JSONResponse(body, status_code=status)
    except PresenceHistoryInvalid as exc:
        return _presence_error(str(exc), 400, exc.code, "rejected", field=exc.field)
    except PresenceAdmissionError as exc:
        return _presence_exception(exc, 403 if exc.code == "presence_origin_mismatch" else 409)
    except store.StoreMigrating:  # an older row may be this event's live capture: duplicates are undecided
        return _presence_error("saved dialogue history is still importing; retry the same request", 503,
                               "presence_history_importing", "retry")
    except Exception:
        log.warning("Presence history import failed for skill %s", skill_name, exc_info=True)
        return _presence_error("presence history import unavailable; retry the same request", 503,
                               "presence_history_unavailable", "retry")
    finally:
        ctx._leave_inflight(f"{skill_name}:history")


def _presence_work_view(
    ctx: host.HostServiceContext, skill_name: str, work_ref: str, binding_id: str,
) -> tuple[int, Dict[str, Any]]:
    from ouroboros.task_results import load_task_result

    stored = load_task_result(ctx.data_dir, work_ref) or {}
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
    transport = str(presence.get("transport_skill") or "")
    if not transport and presence.get("binding_id"):
        # Older durable children omitted the connector. Its retained legacy binding is
        # still authentication evidence even when paused or disconnected from Places.
        from ouroboros.presence_bindings import list_presence_bindings
        transport = next((b.transport_skill for b in list_presence_bindings(ctx.data_dir)
                          if b.binding_id == presence["binding_id"]), "")
    if (transport != skill_name or
            (binding_id and binding_id not in {str(presence.get("binding_id") or ""), str(presence.get("place_id") or "")})):
        return 404, {"ok": False, "error": "presence work reference not found",
                     "code": "presence_work_not_found", "disposition": "rejected"}
    from ouroboros.presence_recovery import retained_work_view

    return retained_work_view(stored, work_ref, ctx.data_dir)


async def _api_presence_work(request: Request) -> JSONResponse:
    """Read a correlated late result, or retain an attributed inbox observation."""
    from ouroboros.gateway import host_service as host
    ctx: host.HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except OSError:
        return _presence_error("presence authentication unavailable", 500, "presence_auth_unavailable", "retry")
    work_ref = str(request.path_params.get("work_ref") or "").strip()
    binding_id = str(request.query_params.get("binding_id") or "").strip()
    try:
        if request.method == "POST":
            from ouroboros.presence_observations import record_transport_queue

            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) not in ({"transport_queue"}, {"binding_id", "transport_queue"}):
                raise ValueError("expected transport_queue and optional binding_id")
            binding_id = str(payload.get("binding_id") or "").strip()
            status, body = await host._bounded_host_read(ctx, functools.partial(_presence_work_view, ctx, skill_name, work_ref, binding_id))
            if status >= 400:
                return JSONResponse(body, status_code=status)
            # Ownership was checked against the authenticated connector, including
            # old retained children. The writer binds the actual stored binding.
            body = await asyncio.to_thread(record_transport_queue, ctx.data_dir, work_ref, binding_id,
                                            payload["transport_queue"])
            return JSONResponse(body)
        status, body = await asyncio.to_thread(_presence_work_view, ctx, skill_name, work_ref, binding_id)
    except (ValueError, json.JSONDecodeError) as exc:
        if getattr(exc, "code", ""):
            return _presence_exception(exc, 404)
        return _presence_error(str(exc), 400, "presence_observation_invalid", "rejected")
    except store.StoreMigrating:
        return _presence_error("the task store is importing; retry shortly", 503, "migrating", "retry")
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code:
            return _presence_exception(exc, 404)
        log.debug("Host service presence work lookup failed", exc_info=True)
        return _presence_error("presence work lookup failed", 500, "presence_work_unavailable", "retry")
    return JSONResponse(body, status_code=status)


async def _api_presence_place(request: Request) -> JSONResponse:
    """An authenticated connector's outbox rechecks current state without model work."""
    from ouroboros.gateway import host_service as host
    ctx = request.app.state.host_service_context
    try:
        skill, _token = await host._authenticated(ctx, request.headers.get("x-skill-token", ""), "presence")
        place = await host._bounded_host_read(ctx, functools.partial(
            _outbox_place, ctx.data_dir, connector=skill,
            provider=str(request.query_params.get("provider") or ""),
            account_id=str(request.query_params.get("account_id") or ""),
            conversation_id=str(request.query_params.get("conversation_id") or ""),
            thread_id=str(request.query_params.get("thread_id") or "")))
        if place is None:
            return _presence_error("Connect this place in Consciousness > Places.", 404, "place_not_connected", "blocked")
        return JSONResponse({"place_id": place.place_id, "label": place.label, "state": place.state})
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except (ValueError, OSError) as exc:
        return _presence_error(str(exc), 503, "places_unavailable", "retry")


def _outbox_place(drive_root, *, connector, provider, account_id, conversation_id, thread_id=""):
    from ouroboros.places import places_of_connector, resolve_place

    if not conversation_id:
        restricted = next((p for p in places_of_connector(drive_root, connector)
                           if p.provider == provider and p.account_id in ("", "*", account_id)
                           and p.state in {"silent", "paused"}), None)
        if restricted is not None:
            return restricted
    return resolve_place(drive_root, connector=connector, provider=provider,
                         account_id=account_id, conversation_id=conversation_id, thread_id=thread_id)
