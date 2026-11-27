"""Connector-facing place intake, receipts and correlated work reads.

``presence_intake`` is the one intake after capture for every ingress of a place event: a connector's
``/presence/turn`` and a linked Ouroboros's message (``gateway/peer_letters_api.py``), on the Host context of the
data root (``host_context``): one turn registry and one reservation for both.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import pathlib
import threading
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
_CONTEXTS: dict[str, Any] = {}  # data root -> its Host context, one per root of this process
_EARLY: set[str] = set()  # roots whose context a message made before their Host Service started
_CONTEXTS_LOCK = threading.Lock()


def host_service_context(data_dir: Any, **dependencies: Any) -> Any:
    """The context a starting Host Service serves ``data_dir`` with (``create_host_service_app``): the one a linked
    Ouroboros's message already made while the boot still ran (``host_context``), now with this service's
    ``dependencies``, else a new one. The root never holds two turn registries or two reservations at once."""
    from ouroboros.gateway.host_service import HostServiceContext

    key = str(pathlib.Path(data_dir).resolve())
    with _CONTEXTS_LOCK:
        if key not in _EARLY:
            ctx = _CONTEXTS[key] = HostServiceContext(pathlib.Path(data_dir), **dependencies)
            return ctx
        _EARLY.discard(key)
        ctx = _CONTEXTS[key]
        for name, value in dependencies.items():
            if value is not None:
                setattr(ctx, name, value)
        return ctx


def host_context(data_dir: Any) -> Any:
    """The Host context of ``data_dir``: its Host Service's, else one made now and kept for that root until that
    service starts and takes it over (``host_service_context``): a boot still running, or a process that starts
    none, as in a test. Never a second one beside it."""
    from ouroboros.gateway.host_service import HostServiceContext

    key = str(pathlib.Path(data_dir).resolve())
    with _CONTEXTS_LOCK:
        if key not in _CONTEXTS:
            _CONTEXTS[key] = HostServiceContext(pathlib.Path(data_dir))
            _EARLY.add(key)
        return _CONTEXTS[key]


class PresenceIntakeRefused(Exception):
    """A typed answer instead of a turn: the Host's own words, status, code and recovery facts."""

    def __init__(self, message: str, status: int, code: str, **facts: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.facts = status, code, facts


def presence_intake(ctx: Any, skill_name: str, admission: Any, event: Any, event_payload: Any, captured: dict, *,
                    staged_files: Any = (), continuing: int = 0, on_dispatch: Any = None) -> tuple[Any, Any]:
    """After admission and capture, in one order for every ingress: a settled turn's replay, the queue's custody,
    an admitted retry, captured wakes, attention, then the rate, the reservation and the dispatch. ``(result,
    execution)``: a settled answer, else the started or joined turn whose futures its carrier resolves. Hearing
    precedes allowance, wake and capacity; a typed retry raises (``PresenceIntakeRefused``, ``WakeImportPending``).
    ``on_dispatch(execution)`` runs only when a new ``PresenceTurnExecution`` is reserved and about to dispatch."""
    from ouroboros.conversation_lease import live_author
    from ouroboros.place_intake import (
        ROUTE_TO_MIND, WAKE_IMPORT_PENDING, allowance_attention, author_hears, decide_attention, offer_arrival,
        settle_heard,
    )
    from ouroboros.presence_admission import WakeImportPending, admit_captured_wakes
    from ouroboros.presence_carrier import dispatch_presence_turn
    from ouroboros.presence_runner import (
        _stored_turn, presence_event_identity, presence_turn_replay, presence_turn_task_id,
    )

    root = ctx.data_dir
    turn_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_event_identity(admission.binding_id, event)
    # A settled turn answers from its durable row without queueing behind its conversation.
    result = presence_turn_replay(root, turn_id, event.conversation_key, identity, continuing)
    if result is None and turn_id not in ctx.presence_turns.live():
        # Saved work the queue keeps (a moved turn's park, Resume or restart return), or that no read yet
        # confirms: answered from that custody before any intake, reservation, admission or author.
        _queue_custody_refusal(root, turn_id, identity)
    # A retry of work already admitted belongs to its frozen custody, even if the
    # owner has since switched Off. Never certify a running/unknown attempt silent.
    started = turn_id in ctx.presence_turns.live() or bool(_stored_turn(root, turn_id, identity))
    if not started:
        started = live_author(event.conversation_key, drive_root=root) == turn_id
    admission = admit_captured_wakes(root, admission, event)
    if result is None and admission.place is not None and not started:
        decision, allowance = decide_attention(root, admission, event)
        # The conversation's current author, else its sole reading root, hears it first, wake or background,
        # before any rate, reservation, slot or allowance; its final drain settles what it accepted.
        wake = author_hears(decision, allowance, event)
        handed, recipient = "", ""
        if wake is not None or allowance.get("reason") == WAKE_IMPORT_PENDING:
            handed, recipient = offer_arrival(root, captured, event, wake=wake)
        if handed == ROUTE_TO_MIND and decision == "silent":  # a pending wake still needs its publication
            decision, allowance = allowance_attention(root)
        if handed and handed != ROUTE_TO_MIND:
            result = settle_heard(root, captured, event, handed, recipient=recipient)
        elif decision == "silent":
            result = settle_heard(root, captured, event, allowance["reason"])
        elif decision == "retry":
            if allowance.get("reason") == WAKE_IMPORT_PENDING:
                raise WakeImportPending()
            raise PresenceIntakeRefused("Consciousness allowance is unavailable; retry this event.", 429,
                                        "consciousness_allowance_exhausted", retry_after=allowance["retry_after"])
        else:
            routed = {ROUTE_TO_MIND: True} if handed == ROUTE_TO_MIND else {}
            admission = replace(admission, allowance={**allowance, **routed})
    if result is not None:
        return result, None
    if not ctx.rate_limiter.allow(f"{skill_name}:presence"):
        raise PresenceIntakeRefused("rate limit exceeded", 429, "presence_rate_limited")
    budget = f"{skill_name}:presence"
    # The body runs in a conversation-lane carrier (owner decision 1A): the Host keeps
    # authentication, capture, dedup and this reservation; the carrier takes the gate
    # (version 2 never waits in it: a busy resource is a typed retry) and reports its
    # initial envelope and terminal as frames that resolve the execution's futures. A process
    # with no started pool generation is a physical fact the lane degrades on (2A): the same
    # body on a thread of this process, disclosed in the chat and the vitals, never undisclosed.
    def dispatch(execution: Any) -> None:
        if on_dispatch is not None:
            on_dispatch(execution)
        dispatch_presence_turn(execution, drive_root=root, skill_name=skill_name, admission=admission,
                               event=event, event_payload=event_payload, staged_files=staged_files,
                               gate_wait=continuing < 2)

    execution, _started = ctx.presence_turns.start_or_dispatch(
        turn_id, identity=identity,
        reserve=functools.partial(ctx._enter_inflight, budget),
        release=functools.partial(ctx._leave_inflight, budget),
        dispatch=dispatch,
    )
    if execution is None:
        raise PresenceIntakeRefused("too many in-flight presence requests", 429, "presence_capacity_full")
    return None, execution


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


async def _unofficial_owner_bot(ctx: "host.HostServiceContext", admission: Any, token_payload: Any) -> bool:
    """Whether an event or history meant for an ``owner_bot`` place (the owner's own bot A) comes from anything but an
    Official ``telegram-bot`` release's exact bytes, the ones its token was issued for: the check A's feed doors make
    (``telegram_owner_bot.official_connector``), here before capture. Other places, B's included, read nothing."""
    from ouroboros.gateway import host_service as host_module
    from ouroboros.telegram_owner_bot import is_owner_bot_place, official_connector

    if not is_owner_bot_place(getattr(admission, "place", None)):
        return False
    issued_for = str((token_payload if isinstance(token_payload, dict) else {}).get("content_hash") or "")
    return not await host_module._bounded_host_read(
        ctx, functools.partial(official_connector, ctx.data_dir, content_hash=issued_for))


_UNOFFICIAL = "Only an Official telegram-bot release, unchanged, brings your own bot's groups."


def _capture_connector_arrival(ctx, admission, event, staged_files, turn_id, identity, continuing):
    """An A event waiting in the connector's inbox still needs the current bridge before its first capture."""
    from ouroboros.places import place_records
    from ouroboros.presence_admission import PresenceAdmissionError
    from ouroboros.presence_events import settled_resend
    from ouroboros.presence_runner import capture_presence_event
    from ouroboros.telegram_owner_bot import current_bridge, exact_owner_bot_identity, is_owner_bot_place

    place = admission.place
    if is_owner_bot_place(place):
        bridge = current_bridge()
        period = bridge.intake(event.account_id) if bridge is not None else None
        if period is not None and bridge.connect_allowed(event.account_id, event.conversation_id):
            # The group read can wait on the feed store. Read the exact connection again before capture.
            current = place_records(ctx.data_dir, persist_import=False).places.get(place.place_id)
            if (current is None or current.connected_revision != place.connected_revision
                    or current.provider != event.provider or not exact_owner_bot_identity(
                        current, event.account_id, event.conversation_id, event.thread_id)):
                raise PresenceAdmissionError("presence_origin_mismatch", "connected_revision")
            if current_bridge() is bridge and bridge.intake(event.account_id) == period:
                return capture_presence_event(admission, event, drive_root=ctx.data_dir, staged_files=staged_files), None
        saved = settled_resend(ctx.data_dir, turn_id, event, identity, continuing)
        if saved is not None:
            return None, saved
        raise PresenceIntakeRefused("The owner's Telegram group feed is unavailable; retry this event.", 503,
                                    "telegram_group_feed_unavailable", retry_after=5)
    return capture_presence_event(admission, event, drive_root=ctx.data_dir, staged_files=staged_files), None


async def _capture_with_group_membership(ctx, admission, event, staged_files, turn_id, identity, continuing):
    from ouroboros.gateway import host_service as host
    from ouroboros.telegram_owner_bot import current_bridge

    capture = functools.partial(_capture_connector_arrival, ctx, admission, event, staged_files, turn_id,
                                identity, continuing)
    try:
        return await host._bounded_host_read(ctx, capture)
    except PresenceIntakeRefused as exc:
        if exc.code != "telegram_group_feed_unavailable" or (bridge := current_bridge()) is None:
            raise
        place = admission.place
        binding = {"account_id": event.account_id, "place_id": place.place_id,
                   "connected_revision": place.connected_revision, "thread_id": event.thread_id}
        if not await bridge.refresh_membership(binding, event.conversation_id) or current_bridge() is not bridge:
            raise
        return await host._bounded_host_read(ctx, capture)


def _legacy_outbox_place(root, connector, target):
    from ouroboros.places import outbox_place
    from ouroboros.telegram_owner_bot import owner_bot_of

    return outbox_place(root, connector=connector, owner_bot=owner_bot_of(root, connector), **target)


def presence_disposition(exc: Exception) -> str:
    """What the sender of an event a typed error answered does next: ``retry`` (the same event again), ``rejected``,
    ``blocked`` or ``retained`` (the runner holds the durable attempt: nothing to resend)."""
    code = str(getattr(exc, "code", "presence_admission_failed"))
    disposition = "blocked"
    if code in {"places_unreadable", "chat_log_unwritable", "presence_result_missing", "presence_bindings_unreadable",
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
    return disposition


def _presence_exception(exc: Exception, status: int) -> JSONResponse:
    code = str(getattr(exc, "code", "presence_admission_failed"))
    disposition = presence_disposition(exc)
    if code == "places_unreadable":
        status = 503
    facts = {"field": getattr(exc, "field", "presence")}
    if getattr(exc, "turn_ref", ""):
        facts["turn_ref"] = exc.turn_ref
        facts["delivery_reporting_version"] = getattr(exc, "delivery_reporting_version", 0)
    if getattr(exc, "retry_after", None) is not None:
        facts["retry_after"] = exc.retry_after
    if getattr(exc, "work_ref", ""):
        facts["work_ref"] = exc.work_ref
    if getattr(exc, "custody", ""):
        facts["custody"] = exc.custody  # ``queued`` or ``unknown``: the queue's custody, never a liveness claim
    manifest = getattr(exc, "attachment_manifest", None)
    if isinstance(manifest, list):
        facts["attachment_manifest"] = [dict(row) for row in manifest if isinstance(row, dict)]
    return _presence_error(str(exc), status, code, disposition, **facts)


def _queue_custody_refusal(data_dir: Any, turn_id: str, identity: str) -> None:
    """Raise the typed retained answer of a turn whose saved work the queue keeps (``presence_recovery.queue_custody``),
    before any intake, reservation, admission or author; a canonical terminal is the replay's, not this answer."""
    from pathlib import Path

    from ouroboros.presence_recovery import CUSTODY_HOLDS, custody_refusal, queue_custody
    from ouroboros.presence_runner import _retry_target, _stored_turn

    physical = _retry_target(Path(data_dir), turn_id, identity) if identity else turn_id
    stored = _stored_turn(Path(data_dir), physical, identity)
    if stored and (custody := queue_custody(Path(data_dir), physical, stored)) in CUSTODY_HOLDS:
        raise custody_refusal(stored, physical, custody)


def _admit_presence(ctx: host.HostServiceContext, skill_name: str, binding_id: str, *, live_event: bool = True,
                    persist_import: bool = True, **origin: str) -> Any:
    from ouroboros.presence_admission import admit_presence_turn

    return admit_presence_turn(
        drive_root=ctx.data_dir,
        authenticated_transport_skill=skill_name,
        binding_id=binding_id,
        live_event=live_event,
        persist_import=persist_import,
        **origin,
    )


async def _api_presence_turn(request: Request) -> JSONResponse:
    """Capture a connector event and run its matched focus under consciousness authority.

    Authentication, admission and file confinement run off the event loop. The turn is
    host work this request only waits on (``presence_runner.PresenceTurnExecutions``): it
    queues on the gate as a coroutine and runs on its own thread. Every version releases
    the request reservation at a qualified park (or terminal settlement); version 1
    returns the initial envelope while version 0's HTTP waiter stays open until the
    terminal result, or the typed retained answer of a moved turn the queue parks. Version 2 takes
    that envelope as soon as its author accepts intake,
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
    from ouroboros.places import PlaceError
    from ouroboros.presence_admission import PresenceAdmissionError, WakeImportPending
    from ouroboros.presence_continuation import continuation_version, turn_response
    from ouroboros.presence_events import (
        PresenceEventInvalid,
        connector_fact_labels,
        outside_origin,
        parse_event_wire,
        settled_resend,
    )
    from ouroboros.presence_executions import PresenceTurnInterrupted
    from ouroboros.presence_runner import (
        PresenceTurnError,
        PresenceTurnNotStarted,
        presence_turn_task_id,
        presence_event_identity,
    )

    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) - {
            "binding_id", "event", "staged_files", "delivery_reporting_version", "continuation_version",
            "attention_version", "connected_revision",
        }:
            return _presence_error("invalid presence payload", 400, "presence_payload_invalid", "rejected")
        # The connection an owner_bot event arrived under (telegram_owner_bot); -1 when the sender named none.
        revision = payload.get("connected_revision", -1)
        revision = revision if type(revision) is int and revision >= 0 else -1
        reporting_version = delivery_reporting_version(payload.get("delivery_reporting_version", 0))
        continuing = continuation_version(payload.get("continuation_version", 0))
        from ouroboros.place_intake import attention_version
        attending = attention_version(payload.get("attention_version", 0))  # negotiation, never event identity
        event_payload = payload.get("event")
        # Check the wire and resolve its origin without writing a legacy place.
        binding_id = str(payload.get("binding_id") or "")
        event = await host._bounded_host_read(ctx, functools.partial(
            parse_event_wire, event_payload,
            staged_files=payload.get("staged_files"), reporting_version=reporting_version, continuing=continuing))
        admission = await host._bounded_host_read(ctx, functools.partial(
            host._admit_presence, ctx, skill_name, binding_id, persist_import=False,
            provider=event.provider, account_id=event.account_id, conversation_id=event.conversation_id,
            thread_id=event.thread_id, connected_revision=revision))
        if outside_origin(admission, event):
            return _presence_error("presence event does not match its owner-created binding", 403,
                                   "presence_origin_mismatch", "rejected")
        if await _unofficial_owner_bot(ctx, admission, _token_payload):  # before any capture, as at A's feed doors
            return _presence_error(_UNOFFICIAL, 403, "connector_mismatch", "blocked")
        staged_files = await host._bounded_host_read(
            ctx, functools.partial(host._presence_staged_files, ctx, skill_name, payload.get("staged_files")))
        turn_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
        identity = presence_event_identity(admission.binding_id, event)
        try:
            declared = await host._bounded_host_read(ctx, functools.partial(
                connector_fact_labels, ctx.data_dir, admission, event))
        except (PresenceEventInvalid, WakeImportPending):
            # A resend matching a settled arrival's retained input keeps its saved answer (with negotiated attention)
            # after retiring a signal; other events meet ``settled_resend``'s publication check.
            settled = await host._bounded_host_read(ctx, functools.partial(
                settled_resend, ctx.data_dir, turn_id, event, identity, continuing))
            if settled is None:
                raise
            return JSONResponse(await host._bounded_host_read(ctx, functools.partial(
                turn_response, settled, continuing, ctx.data_dir, attending)))
        admission = replace(admission, fact_labels=declared)
        # Hearing precedes allowance, wake and capacity; cached custody remains the replay authority.
        captured, saved = await _capture_with_group_membership(ctx, admission, event, staged_files, turn_id,
                                                               identity, continuing)
        if saved is not None:
            return JSONResponse(await host._bounded_host_read(ctx, functools.partial(
                turn_response, saved, continuing, ctx.data_dir, attending)))
        result, execution = await host._bounded_host_read(ctx, functools.partial(
            presence_intake, ctx, skill_name, admission, event, event_payload, captured,
            staged_files=staged_files, continuing=continuing))
        if execution is not None:
            # Versions 1 and 2 take a continuing author's initial envelope; v0 waits for its terminal.
            result = await asyncio.wrap_future(execution.initial if continuing else execution.result)
        return JSONResponse(await host._bounded_host_read(ctx, functools.partial(
            turn_response, result, continuing, ctx.data_dir, attending)))
    except PresenceIntakeRefused as exc:
        return _presence_error(str(exc), exc.status, exc.code, "retry", **exc.facts)
    except json.JSONDecodeError:
        return _presence_error("invalid json", 400, "presence_payload_invalid", "rejected")
    except PresenceEventInvalid as exc:
        return _presence_error(exc.message, 400, exc.code, "rejected")
    except WakeImportPending as exc:
        return _presence_error(str(exc), 429, exc.code, "retry", retry_after=exc.retry_after)
    except (PresenceAdmissionError, PresenceTurnError) as exc:
        return _presence_exception(exc, 403 if exc.code == "presence_origin_mismatch"
                                   else 429 if exc.code in {"presence_capacity_full", WakeImportPending.code} else 409)
    except PresenceTurnNotStarted as exc:
        return _presence_error(str(exc), 503, "presence_turn_not_started", "retry")
    except PresenceTurnInterrupted as exc:  # its carrier ended unreported: the resend reads the durable row
        return _presence_error(str(exc), 503, "presence_turn_interrupted", "retry")
    except OSError as exc:
        return _presence_error(str(exc), 400, "presence_io_unavailable", "retry")
    except PlaceError as exc:  # a place read after capture (attention's current choices) answers as admission's does
        return _presence_exception(exc, 409)
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
    from ouroboros.presence_events import outside_origin
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
            host._admit_presence, ctx, skill_name, parsed.binding_id, live_event=False, persist_import=False,
            provider=first.provider, account_id=first.account_id, conversation_id=first.conversation_id,
            thread_id=first.thread_id))
        if outside_origin(admission, first):
            return _presence_error("presence history does not match its owner-created binding", 403,
                                   "presence_origin_mismatch", "rejected")
        if await _unofficial_owner_bot(ctx, admission, _token_payload):
            return _presence_error(_UNOFFICIAL, 403, "connector_mismatch", "blocked")
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
    ctx: host.HostServiceContext, skill_name: str, work_ref: str, binding_id: str, attention: int = 0,
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
    from ouroboros.place_intake import attention_view
    from ouroboros.presence_recovery import retained_work_view

    status, body = retained_work_view(stored, work_ref, ctx.data_dir)  # a negotiated read adds what the row proves
    return status, ({**body, "attention": attention_view(None, stored)} if attention else body)


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
        from ouroboros.place_intake import attention_version
        try:
            attending = attention_version(int(request.query_params.get("attention_version") or 0))
        except ValueError as exc:
            return _presence_error(str(exc), 400, "presence_payload_invalid", "rejected")
        status, body = await asyncio.to_thread(_presence_work_view, ctx, skill_name, work_ref, binding_id, attending)
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
        operation = request.query_params.get("operation", "")
        if operation:
            from ouroboros.place_guard import outbox_binding_facts
            from ouroboros.places import MATCH_SCOPE, MATCHES
            match = request.query_params.get("match", MATCH_SCOPE)  # absent: the previous scope lookup
            if (operation not in {"bind", "check"} or match not in MATCHES
                    or request.query_params.get("require_place", "false") not in {"true", "false"}):
                return _presence_error("Invalid place binding operation.", 400, "place_binding_invalid", "rejected")
            try:
                binding = json.loads(request.query_params.get("binding", "null")) if operation == "check" else None
                if operation == "check" and not isinstance(binding, dict):
                    raise ValueError("A retained binding object is required for delivery checks.")
                target = {key: request.query_params.get(key, "") for key in
                          ("provider", "account_id", "conversation_id", "thread_id")}
                result = await host._bounded_host_read(ctx, functools.partial(
                    outbox_binding_facts, ctx.data_dir, skill, target, binding=binding,
                    require_place=request.query_params.get("require_place") == "true", match=match,
                    contribution_id=request.query_params.get("contribution_id", "")))
                return JSONResponse({"ok": True, **result})
            except ValueError as exc:
                from ouroboros.places import PlaceError
                if isinstance(exc, PlaceError):
                    raise
                return _presence_error(str(exc), 400, "place_binding_invalid", "rejected")
        target = {key: str(request.query_params.get(key) or "")
                  for key in ("provider", "account_id", "conversation_id", "thread_id")}
        place = await host._bounded_host_read(ctx, functools.partial(
            _legacy_outbox_place, ctx.data_dir, skill, target))
        if place is None:
            return _presence_error("Add this place in Circles.", 404, "place_not_connected", "blocked")
        return JSONResponse({"place_id": place.place_id, "label": place.label, "state": place.state})
    except host.HostServiceAuthError as exc:
        return _presence_error(str(exc), 403, "presence_auth_blocked", "blocked")
    except (ValueError, OSError) as exc:
        # The words a connector's Held row repeats; the exception stays a diagnostic beside them.
        return _presence_error("Ouroboros could not read whether this place allows sending right now.", 503,
                               "places_unavailable", "retry", detail=str(exc))
