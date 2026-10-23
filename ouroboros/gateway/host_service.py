"""Loopback-only Host Service API for privileged skill callbacks."""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import hmac
import json
import logging
import math
import os
import pathlib
import threading
import time
from collections import defaultdict, deque
from contextlib import ExitStack
from typing import Any, Callable, Deque, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros.contracts.chat_id_policy import A2A_CHAT_ID_MAX, A2A_CHAT_ID_MIN, is_a2a_chat_id
from ouroboros.event_bus import get_global_event_bus
from ouroboros.config import WS_RELAY_BURST, WS_RELAY_REFILL_PER_SEC
from ouroboros.gateway._helpers import run_sync_to_completion
from ouroboros.server_control import dispatch_accepted_restart
from ouroboros import chat_uploads
from ouroboros.chat_uploads import store_upload
from ouroboros.presence_delivery import (
    DELIVERY_VERSION, PresenceDeliveryRecorder,
)
from ouroboros.skill_loader import (
    find_skill,
    grant_status_for_skill,
    load_enabled,
)
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso
from ouroboros import store

log = logging.getLogger(__name__)
_json_error = lambda message, status=500: JSONResponse({"ok": False, "error": message}, status_code=status)

DEFAULT_HOST_SERVICE_HOST = "127.0.0.1"
DEFAULT_HOST_SERVICE_PORT = 8767
AUTH_TOKEN_FILENAME = "auth_token.json"
# /identity advertises POST /notify for side-effect-free compatibility checks.
NOTIFY_VERSION = 1

# Owner decision 2026-09-06: a burst must not silence a widget. WS relay alone
# uses a 60-message token bucket refilling at 1/s; other lanes keep sliding windows.


class HostServiceAuthError(Exception):
    """Raised when a skill token cannot be authenticated."""


class _RateLimiter:
    """The one Host Service admission limiter, two policies under one lock.

    ``allow(key)`` is the sliding window (``limit`` hits per ``window_sec``) the
    chat/presence/decision/tools lanes keep unchanged. ``allow_burst(key)`` is a
    token bucket for the WS relay lane only: it also AGGREGATES its refusals per
    key so a burst is reported once (first refusal, then one summary when the
    lane admits again or the bucket goes idle) instead of one log line per
    dropped message; ``on_burst_end(key, dropped, duration_sec)`` is the sink.
    """

    def __init__(
        self,
        limit: int = 60,
        window_sec: float = 60.0,
        *,
        on_burst_end: Optional[Callable[[str, int, float], None]] = None,
    ):
        self.limit = limit
        self.window_sec = window_sec
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        # Token buckets: key -> [tokens, last_refill, capacity, refill_per_sec].
        self._buckets: Dict[str, list] = {}
        # Refusals since the last admit: key -> {"dropped", "since", "last"}.
        self._refused: Dict[str, Dict[str, float]] = {}
        self._on_burst_end = on_burst_end
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> list:
        # Drop keys idle past the window so _hits does not grow unbounded as
        # distinct skill keys ({skill}:{endpoint}) churn over the process
        # lifetime. Must pop each key's stale timestamps FIRST, then delete the
        # ones left empty (an idle key still holds stale, un-popped entries).
        # Collect-then-delete avoids mutating the dict during iteration.
        # Caller holds self._lock. Returns the refusal bursts whose bucket went
        # idle without a later admit, for the caller to report off the lock.
        stale = []
        for key, hits in self._hits.items():
            while hits and now - hits[0] > self.window_sec:
                hits.popleft()
            if not hits:
                stale.append(key)
        for key in stale:
            del self._hits[key]
        ended = []
        for key, bucket in list(self._buckets.items()):
            tokens, last, capacity, rate = bucket
            if min(capacity, tokens + (now - last) * rate) >= capacity:
                del self._buckets[key]
                burst = self._refused.pop(key, None)
                if burst:
                    ended.append((key, burst))
        return ended

    def _report(self, ended: list) -> None:
        for key, burst in ended:
            if self._on_burst_end is None:
                continue
            try:
                self._on_burst_end(key, int(burst["dropped"]), float(burst["last"] - burst["since"]))
            except Exception:
                log.debug("Rate-limiter burst sink failed for %s", key, exc_info=True)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        ended: list = []
        with self._lock:
            # Amortized cleanup: at most once per window, under the existing lock.
            if now - self._last_sweep > self.window_sec:
                ended = self._sweep(now)
                self._last_sweep = now
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_sec:
                hits.popleft()
            if len(hits) >= self.limit:
                admitted = False
            else:
                hits.append(now)
                admitted = True
        self._report(ended)
        return admitted

    def allow_burst(
        self,
        key: str,
        *,
        capacity: int = WS_RELAY_BURST,
        refill_per_sec: float = WS_RELAY_REFILL_PER_SEC,
    ) -> Dict[str, Any]:
        """Token-bucket admission for one key.

        Returns ``{"allowed", "retry_after_sec", "dropped_in_burst"}``: on a
        refusal ``retry_after_sec`` is the wait until one token exists and
        ``dropped_in_burst`` counts this refusal and every earlier one since the
        last admit (``1`` marks the first refusal of a burst).
        """
        now = time.monotonic()
        ended: list = []
        with self._lock:
            if now - self._last_sweep > self.window_sec:
                ended = self._sweep(now)
                self._last_sweep = now
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = [float(capacity), now, float(capacity), float(refill_per_sec)]
            tokens = min(bucket[2], bucket[0] + (now - bucket[1]) * bucket[3])
            bucket[1] = now
            if tokens >= 1.0:
                bucket[0] = tokens - 1.0
                burst = self._refused.pop(key, None)
                if burst:
                    ended.append((key, burst))
                verdict = {"allowed": True, "retry_after_sec": 0.0, "dropped_in_burst": 0}
            else:
                bucket[0] = tokens
                burst = self._refused.setdefault(key, {"dropped": 0, "since": now, "last": now})
                burst["dropped"] += 1
                burst["last"] = now
                verdict = {
                    "allowed": False,
                    "retry_after_sec": (1.0 - tokens) / bucket[3],
                    "dropped_in_burst": int(burst["dropped"]),
                }
        self._report(ended)
        return verdict


class HostServiceContext:
    """Mutable host-service dependencies kept injectable for tests."""

    def __init__(
        self,
        data_dir: pathlib.Path,
        *,
        bridge_getter: Optional[Callable[[], Any]] = None,
        tool_schemas_getter: Optional[Callable[[], list[dict[str, Any]]]] = None,
        ws_broadcaster_getter: Optional[Callable[[], Callable[[dict], None]]] = None,
        presence_runner: Optional[Callable[..., Any]] = None,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.bridge_getter = bridge_getter or self._default_bridge
        self.tool_schemas_getter = tool_schemas_getter or self._default_tool_schemas
        self.ws_broadcaster_getter = ws_broadcaster_getter or self._default_ws_broadcaster
        self.presence_runner = presence_runner or self._default_presence_runner
        self.rate_limiter = _RateLimiter(on_burst_end=self._ws_relay_burst_ended)
        self._inflight: Dict[str, int] = defaultdict(int)
        self._inflight_lock = threading.Lock()
        # Auth and pre-turn Presence reads may both precede turn reservation.
        # Bound their combined occupancy of the shared default executor; queued
        # requests hold no thread, and physical completion returns the permit.
        self._auth_capacity = threading.BoundedSemaphore(2)
        self._counter_lock = threading.Lock()
        self.presence_deliveries = PresenceDeliveryRecorder(self.data_dir)
        from ouroboros.presence_runner import PresenceTurnExecutions

        self.presence_turns = PresenceTurnExecutions()

    def _ws_relay_burst_ended(self, key: str, dropped: int, duration_sec: float) -> None:
        """Report one aggregated WS relay refusal burst: a warning plus one
        durable ``host_service_ws_relay_dropped`` row (the ``broadcast_partial_failure``
        precedent in ``gateway/ws.py``), never one line per dropped message."""
        skill = key.rsplit(":", 1)[0]
        log.warning(
            "Host Service WS relay for skill %r dropped %d message(s) over %.1fs "
            "(burst reserve %d, refill %.0f/s)",
            skill, dropped, duration_sec, WS_RELAY_BURST, WS_RELAY_REFILL_PER_SEC,
        )
        try:
            append_jsonl(self.data_dir / "logs" / "events.jsonl", {
                "ts": utc_now_iso(),
                "type": "host_service_ws_relay_dropped",
                "skill": skill,
                "dropped": int(dropped),
                "duration_sec": round(float(duration_sec), 3),
            })
        except Exception:
            log.debug("Failed to record host_service_ws_relay_dropped event", exc_info=True)

    def _default_bridge(self) -> Any:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is None:
            raise RuntimeError("message bridge is not initialized")
        return bridge

    def _default_tool_schemas(self) -> list[dict[str, Any]]:
        try:
            from supervisor.workers import REPO_DIR
            from ouroboros.tools.registry import ToolRegistry

            return list(ToolRegistry(pathlib.Path(REPO_DIR), self.data_dir).schemas())
        except Exception:
            log.debug("Host service could not read tool schemas", exc_info=True)
            return []

    def _default_ws_broadcaster(self) -> Callable[[dict], None]:
        from ouroboros.gateway.ws import broadcast_ws_sync

        return broadcast_ws_sync

    def _default_presence_runner(self, **kwargs: Any) -> Any:
        from ouroboros.presence_runner import run_presence_turn
        from supervisor.workers import REPO_DIR, get_event_q

        return run_presence_turn(
            repo_dir=pathlib.Path(REPO_DIR),
            drive_root=self.data_dir,
            event_queue=get_event_q(),
            **kwargs,
        )

    async def admit_presence_gate(self, conversation_key: str) -> Any:
        """Coroutine admission into the configured presence gate; the lease the turn runs under."""
        from ouroboros.presence_runner import admit_configured_gate

        return await admit_configured_gate(self.data_dir, conversation_key)

    @property
    def skills_state_dir(self) -> pathlib.Path:
        return self.data_dir / "state" / "skills"

    def authenticate_token_payload(self, raw_token: str) -> tuple[str, Dict[str, Any]]:
        token = str(raw_token or "").strip()
        if not token:
            raise HostServiceAuthError("missing skill token")
        root = self.skills_state_dir
        if not root.exists():
            raise HostServiceAuthError("no skill tokens are registered")
        for skill_dir in root.iterdir():
            if not skill_dir.is_dir():
                continue
            payload = read_json_dict(skill_dir / AUTH_TOKEN_FILENAME) or {}
            expected = str(payload.get("token") or "")
            if expected and hmac.compare_digest(expected, token):
                self._assert_active_token(skill_dir.name, payload)
                return skill_dir.name, payload
        raise HostServiceAuthError("invalid skill token")

    def _assert_active_token(self, skill_name: str, token_payload: Dict[str, Any]) -> None:
        loaded = find_skill(self.data_dir, skill_name)
        if loaded is None:
            raise HostServiceAuthError(f"skill {skill_name!r} is not installed")
        if not loaded.review.gate_for(loaded.content_hash)["executable_review"]:
            raise HostServiceAuthError(f"skill {skill_name!r} has no current executable review authority")
        if not load_enabled(self.data_dir, skill_name):
            raise HostServiceAuthError(f"skill {skill_name!r} is disabled")
        if str(token_payload.get("content_hash") or "") != str(loaded.content_hash or ""):
            raise HostServiceAuthError(f"skill {skill_name!r} token is stale")

    def require_permission(self, skill_name: str, token_payload: Dict[str, Any], permission: str) -> None:
        loaded = find_skill(self.data_dir, skill_name)
        if loaded is not None:
            status = grant_status_for_skill(self.data_dir, loaded)
            granted = set(status.get("granted_permissions") or [])
        else:
            raise HostServiceAuthError(f"skill {skill_name!r} is not installed")
        if permission.startswith("subscribe_event:"):
            topic = permission.split(":", 1)[1]
            declared = set(str(item or "").strip() for item in (loaded.manifest.subscribe_events or []))
            permissions = set(str(item or "").strip() for item in (loaded.manifest.permissions or []))
            if topic == "skill.lifecycle" and "subscribe_event" in permissions and topic in declared:
                return
        if permission not in granted:
            raise HostServiceAuthError(f"skill {skill_name!r} lacks grant {permission!r}")

    def _enter_inflight(self, skill_name: str, limit: int = 5) -> bool:
        with self._inflight_lock:
            current = self._inflight[skill_name]
            if current >= limit:
                return False
            self._inflight[skill_name] = current + 1
            return True

    def _leave_inflight(self, skill_name: str) -> None:
        with self._inflight_lock:
            self._inflight[skill_name] = max(0, self._inflight[skill_name] - 1)

    def allocate_internal_chat_id(self, skill_name: str, range_name: str) -> int:
        if range_name != "a2a":
            raise ValueError("unsupported internal chat id range")
        counter_path = self.skills_state_dir / skill_name / "chat_id_counter.json"
        with self._counter_lock:
            data = read_json_dict(counter_path) or {}
            next_id = int(data.get("next_chat_id") or A2A_CHAT_ID_MAX)
            chat_id = next_id
            if chat_id < A2A_CHAT_ID_MIN:
                chat_id = A2A_CHAT_ID_MAX
            atomic_write_json(
                counter_path,
                {
                    "range_name": range_name,
                    "last_chat_id": chat_id,
                    "next_chat_id": chat_id - 1,
                    "updated_at": utc_now_iso(),
                },
            )
            return chat_id


async def _bounded_host_read(ctx: HostServiceContext, operation: Callable[[], Any]) -> Any:
    """Keep pre-reservation disk work off-loop without filling the shared executor."""
    while not ctx._auth_capacity.acquire(blocking=False):
        await asyncio.sleep(0.01)
    try:
        work = asyncio.get_running_loop().run_in_executor(None, operation)
    except BaseException:
        ctx._auth_capacity.release()
        raise
    # A disconnected caller must not release a permit while its worker still runs.
    work.add_done_callback(lambda _done: ctx._auth_capacity.release())
    return await asyncio.shield(work)


async def _authenticated(
    ctx: HostServiceContext, raw_token: str, permission: str = "",
) -> tuple[str, Dict[str, Any]]:
    """Resolve the request's skill (and one grant) off the event loop.

    Token discovery reads every registered skill's token, review, enablement and grants
    from disk; inline, one slow walk stalled every request on the loop the owner's API
    shares. It is short read-only work, so the default executor serves it: Presence turns
    no longer wait there (``presence_runner.PresenceTurnExecutions``).
    """

    def resolve() -> tuple[str, Dict[str, Any]]:
        skill_name, token_payload = ctx.authenticate_token_payload(raw_token)
        if permission:
            ctx.require_permission(skill_name, token_payload, permission)
        return skill_name, token_payload

    return await _bounded_host_read(ctx, resolve)


def _token_from_websocket(websocket: WebSocket) -> str:
    header = websocket.headers.get("x-skill-token", "")
    if header:
        return header
    for protocol in websocket.scope.get("subprotocols") or []:
        text = str(protocol or "")
        prefix = "ouroboros.host.events.v1."
        if text.startswith(prefix):
            return text[len(prefix):]
    return ""


def _identity_facts(ctx: HostServiceContext) -> tuple[str, str]:
    identity_path = ctx.data_dir / "memory" / "identity.md"
    name = "Ouroboros"
    description = ""
    try:
        if identity_path.exists():
            lines = identity_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if line.startswith("# "):
                    name = line.lstrip("# ").strip() or name
                    continue
                if line.strip() and not description:
                    description = line.strip()
                    break
    except Exception:
        log.debug("Failed to read identity for host service", exc_info=True)
    return name, description


async def _api_identity(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        await _authenticated(ctx, request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    name, description = await asyncio.to_thread(_identity_facts, ctx)
    from ouroboros.places import WAKE_FACTS
    from ouroboros.presence_continuation import CONTINUATION_VERSION
    from ouroboros.presence_owner_contact import MAIN_CONTACT_VERSION
    from ouroboros.presence_recovery import WORK_OUTPUT_VERSION

    return JSONResponse({"ok": True, "name": name, "description": description,
                         "presence_delivery_version": DELIVERY_VERSION, "places_version": 1,
                         "presence_continuation_version": CONTINUATION_VERSION,
                         "presence_main_contact_version": MAIN_CONTACT_VERSION,
                         "presence_work_output_version": WORK_OUTPUT_VERSION,
                         "presence_wake_facts": list(WAKE_FACTS),
                         "notify_version": NOTIFY_VERSION})


async def _api_tool_schemas(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:tools"):
        return _json_error("rate limit exceeded", 429)
    schemas = await asyncio.to_thread(ctx.tool_schemas_getter)
    return JSONResponse({"ok": True, "tools": schemas})


async def _api_allocate_internal(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    try:
        payload = await request.json()
        chat_id = await run_sync_to_completion(
            ctx.allocate_internal_chat_id, skill_name, str(payload.get("range_name") or "a2a"),
        )
    except Exception as exc:
        return _json_error(str(exc), 400)
    return JSONResponse({"ok": True, "chat_id": chat_id})


async def _api_chat_inject(request: Request) -> JSONResponse:
    """Inject one owner-channel message; correlate it when the caller names it.

    ``client_message_id`` is the inbound message identity (#667): the host keeps
    it on the canonical inbound row it already writes, answers with the
    correlated ``operation_ref`` on 202/200/504, and a repeated delivery of the
    SAME message (same id, same text, same skill) REJOINS the accepted operation
    instead of enqueueing a second one; a different message under a reused id
    is refused (409), never mistaken for a replay. Without an id the historical
    envelope is byte-identical.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    try:
        payload = await request.json()
        panic_text = str(payload.get("text") or payload.get("image_caption") or "")
        is_panic = panic_text.strip().lower().startswith("/panic")
        from ouroboros.panic_source import source_evidence
        from ouroboros.server_control import request_panic
        if is_panic and request_panic(
            ctx.bridge_getter(), panic_text, source=f"skill:{skill_name}",
            user_id=int(payload.get("user_id") or 0), chat_id=int(payload.get("chat_id") or 0),
            evidence=source_evidence(request, "telegram" if skill_name == "telegram" else "host_service", skill_name=skill_name),
        ):
            return JSONResponse({"ok": True, "status": "accepted"}, status_code=202)
        if is_panic:
            return _json_error("Panic emergency owner unavailable or sender not authorized.", 503)
    except json.JSONDecodeError:
        return _json_error("invalid json", 400)
    except Exception as exc:
        return _json_error(str(exc), 500)
    if not ctx.rate_limiter.allow(f"{skill_name}:inject"):
        return _json_error("rate limit exceeded", 429)
    if not ctx._enter_inflight(skill_name):
        return _json_error("too many in-flight inject requests", 429)
    subscription_id = ""
    pending_uploads = ExitStack()
    try:
        from ouroboros.task_status import SETTLED_STATUSES

        if not isinstance(payload.get("reply_to", ""), str):
            return _json_error("reply_to must be an address string", 400)
        text = str(payload.get("text") or "")
        image_caption = str(payload.get("image_caption") or "")
        client_message_id = str(payload.get("client_message_id") or "").strip()[:128]
        chat_id = int(payload.get("chat_id") or 0)
        if _inline_image_too_large(payload):  # refused on its encoded length, before any decode or queue
            return _json_error(f"image_base64: at most {_INLINE_IMAGE_MAX // (1024 * 1024)} MiB per image", 413)
        wait_for_response = bool(payload.get("wait_for_response", False))
        if wait_for_response and not is_a2a_chat_id(chat_id):
            # A response subscription resolves on the FIRST non-progress frame
            # in the chat. On a human/project chat that frame can be any
            # concurrent task's answer — and, now that owner sends deliver
            # live mid-task, any proactive frame. Only A2A-allocated chats
            # (see /chat/allocate) have single-conversation semantics.
            return _json_error(
                "wait_for_response requires an A2A-allocated chat_id "
                "(allocate one via /chat/allocate-internal)", 400,
            )
        timeout = max(1, min(int(payload.get("timeout_sec") or 1800), 1800)) if wait_for_response else 1800
        correlated = {"operation_ref": operation_ref(chat_id, client_message_id)} if client_message_id else {}
        rejoined = False
        if client_message_id:
            rows = await asyncio.to_thread(_chat_rows, ctx, chat_id)
            inbound = _inbound_row(rows, client_message_id)
            if inbound is not None:
                if str(inbound.get("source") or "") != f"skill:{skill_name}":
                    return _json_error("client_message_id is already bound to another source", 409)
                try:  # the same message means the same words AND the same ordered attachment content
                    incoming, inline_image = await run_sync_to_completion(_inject_identity, ctx, skill_name, payload)
                except ValueError as exc:
                    return _json_error(str(exc), 400)
                # Reuse the accepted byte-proven kinds; same_message still checks every incoming digest/name.
                logged = text.strip() or image_caption.strip() or chat_uploads.attachment_placeholder(
                    inline_image, {"chat_attachments": inbound.get("attachments"), "chat_attachment_uploads": incoming})
                if (not chat_uploads.same_message(inbound, logged, incoming)
                        or (inbound.get("reply_to") or {}).get("address", "") != payload.get("reply_to", "")):
                    return _json_error("client_message_id was already used for a different message", 409)
                state = _operation_state(ctx, rows, inbound)
                if state["status"] in SETTLED_STATUSES:
                    return JSONResponse({
                        "ok": True, "response": str(state.get("text") or ""),
                        "status": state["status"], "rejoined": True, **correlated,
                    })
                # A dispatch still owed (the acceptor died before the handoff): the named ingress below
                # hands it over, once; anything else rejoins.
                from supervisor.message_ingress import dispatch_facts

                facts = dispatch_facts(inbound, ctx.data_dir)
                rejoined = facts is None or facts["state"] != "pending"
        metadata: dict[str, Any] = {}
        inline, image_mime = "", ""
        if not rejoined:
            try:
                uploads, refs, image_mime = await run_sync_to_completion(
                    _inject_attachment_uploads, ctx, skill_name, payload, pending_uploads,
                )
            except ValueError as exc:
                return _json_error(str(exc), 400)
            # Vision gets only bytes proven to be an image, under the type they prove; any
            # other inline bytes travel as the ordinary staged file stored above.
            inline = str(payload.get("image_base64") or "") if image_mime else ""
            metadata = {**({"chat_attachment_uploads": uploads} if uploads else {}),
                        **({"chat_attachments": refs} if refs else {})}
        if payload.get("reply_to"):  # no reference adds nothing to a plain inject's kwargs
            metadata["reply_to"] = payload["reply_to"]
        bridge = ctx.bridge_getter()
        response_event: asyncio.Event = asyncio.Event()
        response_holder: dict[str, str] = {}
        if wait_for_response and not client_message_id:
            loop = asyncio.get_running_loop()

            def on_response(response_text: str) -> None:
                response_holder["text"] = response_text
                loop.call_soon_threadsafe(response_event.set)

            subscription_id = bridge.subscribe_response(chat_id, on_response)
        if not rejoined:
            message = dict(
                chat_id=chat_id,
                user_id=int(payload.get("user_id") or 0),
                source=f"skill:{skill_name}",
                sender_label=str(payload.get("sender_label") or skill_name),
                image_base64=inline,
                image_mime=image_mime,
                image_caption=image_caption,
                transport=payload.get("transport") if isinstance(payload.get("transport"), dict) else {},
                **({"task_metadata": metadata} if metadata else {}),
                **({"client_message_id": client_message_id} if client_message_id else {}),
            )
            if client_message_id:
                from supervisor.message_bus import accept_local_message

                try:
                    _, rejoined = await run_sync_to_completion(
                        accept_local_message, bridge, ctx.data_dir, text,
                        retain_inputs=pending_uploads.pop_all,
                        dispatch=functools.partial(dispatch_accepted_restart, bridge), **message,
                    )
                except ValueError as exc:
                    return _json_error(str(exc), 409)
            else:
                # Custody passes BEFORE the handoff: dispatch may enqueue and then raise, so from
                # here a failure is unknown. The copies stay (an orphan at worst), never deleted
                # under queued work; the error still answers the skill, nothing is re-dispatched.
                pending_uploads.pop_all()
                dispatch_accepted_restart(bridge, text, **message)
        if not wait_for_response:
            if rejoined:
                return JSONResponse({"ok": True, "status": "accepted", "rejoined": True, **correlated}, status_code=202)
            return JSONResponse({"ok": True, "status": "queued", **correlated}, status_code=202)
        deadline = time.monotonic() + timeout
        while not response_event.is_set():
            if client_message_id:
                state = await asyncio.to_thread(_owned_operation_state, ctx, skill_name, chat_id, client_message_id)
                if state and state["status"] in {*SETTLED_STATUSES, "lost"}:
                    return JSONResponse({"ok": True, "response": str(state.get("text") or ""),
                                         "status": state["status"], "rejoined": rejoined, **correlated})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The work keeps running; the ref lets the caller recover its
                # late answer through /chat/operations (#667).
                return JSONResponse(
                    {"ok": False, "error": "timed out waiting for response", **correlated},
                    status_code=504,
                )
            try:
                if await request.is_disconnected():
                    return JSONResponse({"ok": False, "error": "client disconnected", **correlated}, status_code=499)
                await asyncio.wait_for(response_event.wait(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue
        return JSONResponse({"ok": True, "response": response_holder.get("text", ""), **correlated})
    except json.JSONDecodeError:
        return _json_error("invalid json", 400)
    except Exception as exc:
        log.debug("Host service chat inject failed", exc_info=True)
        return _json_error(str(exc), 500)
    finally:
        if subscription_id:
            try:
                ctx.bridge_getter().unsubscribe_response(subscription_id)
            except Exception:
                log.debug("Failed to unsubscribe host-service response callback", exc_info=True)
        try:
            pending_uploads.close()
        except OSError:
            log.warning("Could not remove an unaccepted chat upload", exc_info=True)
        finally:
            ctx._leave_inflight(skill_name)


_INJECT_ATTACHMENT_MAX = 25


_INLINE_IMAGE_MAX = 50 * 1024 * 1024  # the routed-staging bound for a transport's inline photo
_B64_BLANKS = " \t\r\n"  # MIME-style line wrapping; any other non-alphabet character is refused
_B64_UNWRAP = str.maketrans("", "", _B64_BLANKS)


def _inline_image_too_large(payload: Dict[str, Any]) -> bool:
    """Whether the inline photo's base64 decodes to more than the bound, judged before any
    decode from its length: the exact decoded size of well-formed base64 (blanks aside,
    ``=`` padding counted); malformed input is refused by the decode itself."""
    encoded = str(payload.get("image_base64") or "")
    tail = encoded.rstrip(_B64_BLANKS)
    padding = 2 if tail.endswith("==") else 1 if tail.endswith("=") else 0
    compact = len(encoded) - sum(encoded.count(blank) for blank in _B64_BLANKS)
    return compact * 3 // 4 - padding > _INLINE_IMAGE_MAX


def _inline_image_bytes(payload: Dict[str, Any]) -> tuple[bytes, bool]:
    """The inline photo's bytes (size preflighted by the caller, re-checked on the decoded
    bytes) and whether they prove an image; line-wrapped base64 is still base64, anything
    else is refused rather than parked or forwarded half-read."""
    raw = str(payload.get("image_base64") or "").translate(_B64_UNWRAP)
    try:
        data = base64.b64decode(raw, validate=True) if raw else b""
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if len(data) > _INLINE_IMAGE_MAX:
        raise ValueError(f"image_base64: at most {_INLINE_IMAGE_MAX // (1024 * 1024)} MiB per image")
    return data, bool(data) and chat_uploads.detect_media(data, "")[1] == "image"


def _skill_state_path(ctx: HostServiceContext, skill_name: str, raw: Any, field: str) -> pathlib.Path:
    """``raw`` resolved, refused unless under the calling skill's OWN state root (a symlink
    that resolves outside is refused)."""
    path = pathlib.Path(str(raw or "")).expanduser().resolve(strict=False)
    try:
        path.relative_to((ctx.skills_state_dir / skill_name).resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{field} is outside this skill's state") from exc
    return path


def _confined_inject_sources(
    ctx: HostServiceContext, skill_name: str, value: Any,
) -> list[tuple[pathlib.Path, str, str]]:
    """``(source, name, mime)`` per ``{path, name?, mime?}``; each a regular file under the
    calling skill's OWN state root (the ``staged_files`` confinement, ``_skill_state_path``)."""
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("attachments must be a list of {path, name?, mime?}")
    if len(value) > _INJECT_ATTACHMENT_MAX:
        raise ValueError(f"attachments: at most {_INJECT_ATTACHMENT_MAX} files per message")
    sources: list[tuple[pathlib.Path, str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"attachments[{index}] must be an object")
        source = _skill_state_path(ctx, skill_name, item.get("path"), f"attachments[{index}]")
        if not source.is_file():
            raise ValueError(f"attachments[{index}] is not a regular file")
        sources.append((source, os.path.basename(str(item.get("name") or "").strip()) or source.name,
                        str(item.get("mime") or "")))
    return sources


def _inject_identity(
    ctx: HostServiceContext, skill_name: str, payload: Dict[str, Any],
) -> tuple[list[dict[str, str]], bool]:
    """The attachment identity this delivery will record (inline bytes first, then files),
    measured from the sources, so a regenerated stored name never changes it; and whether
    the inline bytes are a proven image (what the placeholder text says)."""
    data, inline_image = _inline_image_bytes(payload)
    entries = [chat_uploads.bytes_identity(data, chat_uploads.inline_image_name(data))] if data else []
    entries += [chat_uploads.source_identity(source, name)
                for source, name, _mime in _confined_inject_sources(ctx, skill_name, payload.get("attachments"))]
    return [{"sha256": sha256, "name": name} for sha256, name in entries], inline_image


def _inject_attachment_uploads(
    ctx: HostServiceContext, skill_name: str, payload: Dict[str, Any], cleanup: ExitStack,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Copy a skill's inbound files (#668) and its inline photo into the shared upload store.

    The host copies through the SAME store the browser paperclip uses —
    ``data/uploads``, unique name, verified bytes — so the worker's
    ``stage_task_attachments`` and the secret-name rule see one upload family.
    Returns ``(chat_attachment_uploads specs, measured display refs, the inline photo's
    byte-proven image mime or "")``. Inline bytes are parked ONCE, and the owner sees
    one bubble. The original inline image is always staged, including a solitary
    photo, so later explicit readers retain its pixels. Every lane drops the inline
    copy once uploads stage. Inline bytes that prove no image are no photo: they are
    staged as an ordinary file and never reach vision under the transport's label.
    Each spec carries its ref's measured identity, which staging verifies while copying.
    Each new destination belongs to the request cleanup stack until its message is
    accepted; partial batches and cancelled copy waits therefore cannot orphan earlier copies.
    """
    refs: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    sources = _confined_inject_sources(ctx, skill_name, payload.get("attachments"))

    def park(source: Any, name: str, mime: str = "") -> None:
        stored, ref = store_upload(source, name, data_dir=ctx.data_dir)
        cleanup.callback(stored.unlink, missing_ok=True)
        refs.append(ref)
        specs.append({"path": str(stored), "label": name, "mime": mime or ref["mime"],
                      "size": ref["size"], "sha256": ref["sha256"]})

    data, image = _inline_image_bytes(payload)
    if data:  # parked first: a proven image's ref is refs[0]
        park(data, chat_uploads.inline_image_name(data))
    for source, name, mime in sources:
        park(source, name, mime)
    return specs, refs, refs[0]["mime"] if image else ""


def _presence_staged_files(
    ctx: HostServiceContext,
    skill_name: str,
    value: Any,
) -> tuple[pathlib.Path, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError("staged_files must be a list of paths")
    # Keep the host boundary responsible only for request shape and source
    # confinement.  Missing/non-file inputs and the staging limit belong to
    # the existing canonical staging owner, which emits the complete typed
    # ordinal manifest before Presence can call the model.
    return tuple(_skill_state_path(ctx, skill_name, raw, f"staged_files[{index}]") for index, raw in enumerate(value))


# Compatibility import surface; implementation is owned by the places handler module.
from ouroboros.gateway.host_service_places import (  # noqa: E402
    _admit_presence as _admit_presence,
    _api_presence_delivery as _api_presence_delivery,
    _api_presence_turn as _api_presence_turn,
    _api_presence_work as _api_presence_work,
    _api_presence_place as _api_presence_place,
    _presence_work_view as _presence_work_view,
)


async def _api_chat_decision(request: Request) -> JSONResponse:
    """Relay the owner's answer through the shared owner-decision ingress.

    The SAME ingress as ``POST /api/decisions`` (``task_decision.answer_decision``):
    idempotent per ``request_id``, first answer wins, typed 404/409 refusals.
    ``inject_chat`` is the owner grant that already lets this skill speak as the
    owner's chat input; answering the owner's own quiz needs nothing more (#472).
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:decision"):
        return _json_error("rate limit exceeded", 429)
    try:
        body = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    from ouroboros.gateway.task_decision import answer_decision

    try:
        status, payload = await answer_decision(ctx.data_dir, body, source=f"skill:{skill_name}")
    except Exception as exc:
        log.debug("Host service decision relay failed", exc_info=True)
        return _json_error(str(exc), 500)
    payload.setdefault("ok", status < 400)
    return JSONResponse(payload, status_code=status)


async def _api_ui_language(request: Request) -> JSONResponse:
    """Relay the owner's interface-language choice through the ONE language writer.

    The same seam as ``POST /api/ui/i18n/language`` (``ui_i18n.choose_language``): a tag, a
    language name or a description, resolved and written once for the whole install.
    ``inject_chat`` is the owner grant that already lets this skill speak as the owner's
    chat input; relaying the owner's own ``/language`` command needs nothing more.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:language"):
        return _json_error("rate limit exceeded", 429)
    try:
        body = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    if not isinstance(body, dict):
        return _json_error("request body must be a JSON object", 400)
    from ouroboros.gateway.owner_settings import SettingsDocumentBusy
    from ouroboros.gateway.ui_i18n import choose_language

    def _audit(facts: Dict[str, Any]) -> None:
        append_jsonl(ctx.data_dir / "logs" / "events.jsonl", {
            "ts": utc_now_iso(), "type": "owner_api_action", "action": "ui_language",
            "client_host": f"host_service:{skill_name}", **facts,
        })

    try:
        status, payload = await asyncio.to_thread(choose_language, ctx.data_dir, body, audit=_audit)
    except SettingsDocumentBusy as exc:
        return JSONResponse({"ok": False, "error": str(exc), "code": "settings_busy"}, status_code=503)
    except Exception as exc:
        log.debug("Host service language relay failed", exc_info=True)
        return _json_error(str(exc), 500)
    return JSONResponse(payload, status_code=status)


async def _api_ws_message(request: Request) -> JSONResponse:
    """WS-out bridge: relay a namespaced extension WS event to browser clients.

    Identity is derived from the token (never the body); the host re-derives the ``ext_<len>_<token>_<short>``
    namespace, so a child/companion cannot spoof another skill's events. ``ws_handler`` is a manifest permission,
    not an owner grant (as in-process ``send_ws_message``). A refused relay is a 429 with ``Retry-After`` and
    ``retry_after_sec``/``dropped_in_burst`` in the body.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    loaded = await asyncio.to_thread(find_skill, ctx.data_dir, skill_name)
    if loaded is None:
        return _json_error(f"skill {skill_name!r} is not installed", 403)
    if "ws_handler" not in {str(p).strip() for p in (loaded.manifest.permissions or [])}:
        return _json_error(f"skill {skill_name!r} lacks ws_handler permission", 403)
    verdict = ctx.rate_limiter.allow_burst(f"{skill_name}:ws")
    if not verdict["allowed"]:
        # Visible at the host, aggregated per burst: the first refusal logs once,
        # the rest ride the counter until the bucket admits again (then the
        # context's sink reports the dropped total). The child keeps its
        # best-effort ``None``; ``Retry-After`` says when one token exists.
        if verdict["dropped_in_burst"] == 1:
            log.warning(
                "Host Service WS relay for skill %r refused: burst reserve of %d "
                "messages is empty (refills %.0f/s); further refusals in this burst "
                "are aggregated",
                skill_name, WS_RELAY_BURST, WS_RELAY_REFILL_PER_SEC,
            )
        retry_after = float(verdict["retry_after_sec"])
        return JSONResponse(
            {
                "ok": False,
                "error": "rate limit exceeded",
                "retry_after_sec": round(retry_after, 3),
                "dropped_in_burst": int(verdict["dropped_in_burst"]),
            },
            status_code=429,
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    try:
        payload = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    from ouroboros.extension_loader import extension_surface_name
    from ouroboros.extension_ui_validation import _assert_ws_message_type
    try:
        short = _assert_ws_message_type(str(payload.get("message_type") or ""))
        full = extension_surface_name(skill_name, short)
    except Exception as exc:
        return _json_error(str(exc), 400)
    data = payload.get("data")
    message = {"type": full, "data": dict(data) if isinstance(data, dict) else {}, "skill": skill_name}
    try:
        ctx.ws_broadcaster_getter()(message)
    except Exception:
        log.debug("Host service WS relay broadcast failed", exc_info=True)
        return _json_error("broadcast failed", 500)
    return JSONResponse({"ok": True, "type": full}, status_code=202)


# A2A operation_ref is caller-known "<chat_id>:<client_message_id>" (#667).
# Canonical inbound source, task origin and in-flight registry remain authoritative;
# annotation/outbound IDs only discover candidates. No new record is kept.


def operation_ref(chat_id: int, client_message_id: str) -> str:
    return f"{int(chat_id)}:{client_message_id}"


def _parse_operation_ref(value: Any) -> tuple[int, str]:
    head, sep, tail = str(value or "").partition(":")
    if not sep or not tail.strip():
        raise ValueError("operation_ref must be '<chat_id>:<client_message_id>'")
    return int(head), tail.strip()[:128]


def _chat_rows(ctx: HostServiceContext, chat_id: int) -> list:
    """The chat's retained canonical rows (the dialogue store's room probe, oldest first)."""
    from ouroboros import dialogue_store

    found = dialogue_store.probe_rows(ctx.data_dir, "chat", [("chat_id", int(chat_id))], limit=10_000)
    return [row.row for row in reversed(found)]


def _inbound_row(rows: list, client_message_id: str) -> Optional[Dict[str, Any]]:
    return next(
        (
            row for row in rows
            if str(row.get("direction") or "") == "in"
            and str(row.get("client_message_id") or "") == client_message_id
        ),
        None,
    )


def _operation_state(ctx: HostServiceContext, rows: list, inbound: Dict[str, Any]) -> Dict[str, Any]:
    """Join the existing records into one typed view of an accepted message.

    Authority is the task/turn's complete ingress origin matched to the skill's
    canonical source, then that task's effective retry-aware status; otherwise
    ``pending`` — or ``lost``
    when the host process or session that accepted the message is gone, or this process
    saw its acceptance write raise (the row landed, nothing was queued), and nothing else
    answers. ``cancel_supported`` is true only for work THIS message started
    that the cancellation owner can address: a promoted task or a live direct
    turn; a message steered into a pre-existing task is disclosed, not cancelled.
    """
    from ouroboros.project_dialogue import latest_chat_annotations, entry_matches_source_ref, owner_message_ref_is_valid
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES, load_effective_task_result
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor import queue as task_queue

    chat_id = int(inbound.get("chat_id") or 0)
    client_message_id = str(inbound.get("client_message_id") or "")
    state: Dict[str, Any] = {
        "operation_ref": operation_ref(chat_id, client_message_id),
        "chat_id": chat_id,
        "client_message_id": client_message_id,
        "accepted_at": str(inbound.get("ts") or ""),
        "status": "pending",
        "cancel_supported": False,
    }
    def owns(record: dict) -> bool:
        ref = record.get("origin_message_ref")
        return bool(owner_message_ref_is_valid(ref) and entry_matches_source_ref(inbound, [ref]))

    # An annotation is only a discovery hint. The actual task's complete
    # ingress origin, already scoped to the authenticated skill's row, is the
    # authority. Live queue payloads cover tasks not yet persisted by a worker.
    queued = {}
    cancel_owner_matches = (task_queue.DRIVE_ROOT is not None
                            and pathlib.Path(task_queue.DRIVE_ROOT).resolve() == ctx.data_dir.resolve())
    if cancel_owner_matches:
        with task_queue._queue_lock:
            for task in [*task_queue.PENDING, *(meta.get("task") or {} for meta in task_queue.RUNNING.values())]:
                if owns(task):
                    queued[str(task.get("id") or "")] = dict(task)
    receipt = latest_chat_annotations(ctx.data_dir).get(client_message_id) or {}
    targets = dict.fromkeys([*queued, str(receipt.get("target") or ""), *(str(row.get("task_id") or "") for row in reversed(rows))])
    for target in targets:
        if not target:
            continue
        try:
            raw = queued.get(target) or load_task_result(ctx.data_dir, target, strict=True) or {}
        except (ValueError, store.StoreError):
            continue  # An unreadable/disallowed discovery hint carries no authority.
        if not owns(raw):
            continue
        stored = load_effective_task_result(ctx.data_dir, target, materialize_artifacts=False) or {}
        status = str(stored.get("status") or "")
        state.update({
            "task_id": target,
            "phase": "managed_task",
            "status": status or "pending",
        })
        if status in SETTLED_STATUSES:
            state["text"] = str(stored.get("result") or "")
            # A delivered result can precede post-task work or live descendants.
            # Use the same ownership facts as the existing cancellation ingress.
            state["cancel_supported"] = cancel_owner_matches and (
                task_queue.task_has_live_ownership(target) or task_queue.task_subtree_is_live(target)
            )
        else:
            state["cancel_supported"] = cancel_owner_matches
            if not cancel_owner_matches:
                state["reason"] = "cancel_owner_unavailable"
            if stored.get("cancel_state"):
                state["cancel_state"] = str(stored.get("cancel_state"))
        return state
    for entry in get_direct_activity_registry().snapshot(chat_id):
        activity = get_direct_activity_registry().get(str(entry.get("activity_id") or ""))
        if activity is not None and owns({"origin_message_ref": activity.origin_message_ref}):
            kind = str(entry.get("kind") or "direct_chat")
            state.update({
                "status": "running",
                "phase": kind,
                "task_id": str(entry.get("activity_id") or ""),
                "cancel_supported": kind == "direct_chat" and cancel_owner_matches,
            })
            if kind == "direct_chat" and not cancel_owner_matches:
                state["reason"] = "cancel_owner_unavailable"
            return state
    for row in reversed(rows):
        terminal = str(row.get("task_terminal_status") or "")
        if row.get("direction") in {"out", "system"} and terminal in SETTLED_STATUSES and owns(row):
            state.update({"status": terminal, "text": str(row.get("text") or "")})
            return state
    from supervisor.message_ingress import dispatch_lost, dispatch_taken_over_here

    accepted_session = str(inbound.get("session_id") or "")
    live_session = str((read_json_dict(ctx.data_dir / "state" / "state.json") or {}).get("session_id") or "")
    # A crash keeps the session: the dispatch fact names the ended process (and its queue). A dispatch
    # THIS process took over after the restart (an owed one, by a retry or the recovery) is live.
    ended = dispatch_lost(inbound, ctx.data_dir)
    restarted = accepted_session and live_session and accepted_session != live_session
    if ended or (restarted and not dispatch_taken_over_here(inbound, ctx.data_dir)):
        state.update({"status": "lost", "reason": "host_restarted_before_answer"})
    return state


def _owned_operation_state(
    ctx: HostServiceContext, skill_name: str, chat_id: int, client_message_id: str,
) -> Optional[Dict[str, Any]]:
    """The operation view, or None unless THIS skill injected the message."""
    rows = _chat_rows(ctx, chat_id)
    inbound = _inbound_row(rows, client_message_id)
    if inbound is None or str(inbound.get("source") or "") != f"skill:{skill_name}":
        return None
    return _operation_state(ctx, rows, inbound)


def _cancel_owned_operation(
    ctx: HostServiceContext, skill_name: str, chat_id: int, client_message_id: str, reason: str,
) -> tuple[int, Dict[str, Any]]:
    from ouroboros.task_status import SETTLED_STATUSES

    state = _owned_operation_state(ctx, skill_name, chat_id, client_message_id)
    if state is None:
        return 404, {"ok": False, "error": "operation not found"}
    base = {key: state[key] for key in ("operation_ref", "task_id", "phase", "status") if key in state}
    if state["status"] in SETTLED_STATUSES and not state.get("cancel_supported"):
        return 200, {"ok": True, "outcome": "already_terminal", **base}
    if not state.get("cancel_supported") or not state.get("task_id"):
        reason_code = state.get("reason") or {
            "lost": "host_restarted_before_answer",
        }.get(str(state.get("phase") or state["status"]), "not_started")
        return 409, {"ok": False, "outcome": "cancel_unsupported", "reason": reason_code, **base}
    return _cancel_task_through_owner(skill_name, str(state["task_id"]), reason, base, ctx.data_dir)


def _cancel_task_through_owner(
    skill_name: str, task_id: str, reason: str, base: Dict[str, Any], drive_root: pathlib.Path,
) -> tuple[int, Dict[str, Any]]:
    """The existing cancel ingress shape: durable intent first (fail-closed),
    then the same cascade custody path the browser Stop uses; the typed outcome
    is read back from the effective result, never assumed."""
    from ouroboros.cancel_intents import (
        CancelIntentProjectionCorrupt,
        SCOPE_CASCADE,
        STOP_POLICY_IMMEDIATE,
        request_cancel,
    )
    from ouroboros.gateway.tasks import _run_cascade_cancel
    from ouroboros.task_results import STATUS_CANCELLED
    from ouroboros.task_status import SETTLED_STATUSES, load_effective_task_result
    from supervisor.queue import DRIVE_ROOT, task_has_live_ownership, task_subtree_is_live

    if DRIVE_ROOT is None or pathlib.Path(DRIVE_ROOT).resolve() != drive_root.resolve():
        return 409, {"ok": False, "outcome": "cancel_unsupported", "reason": "cancel_owner_unavailable", **base}

    def _status() -> str:
        stored = load_effective_task_result(pathlib.Path(DRIVE_ROOT), task_id, materialize_artifacts=False) or {}
        return str(stored.get("status") or "")

    if not task_has_live_ownership(task_id) and not task_subtree_is_live(task_id):
        status = _status()
        if status in SETTLED_STATUSES:
            return 200, {"ok": True, "outcome": "already_terminal", **base, "status": status}
        return 503, {"ok": False, "outcome": "unresolved", "reason": "cancellation_did_not_settle", **base, "status": status}
    try:
        request_cancel(
            DRIVE_ROOT, task_id, reason=reason, source=f"skill:{skill_name}",
            scope=SCOPE_CASCADE, allow_settled_target=True,
            requested_stop_policy=STOP_POLICY_IMMEDIATE,
        )
    except CancelIntentProjectionCorrupt:
        log.error("Host Service cancel refused for %s: intent projection corrupt", task_id)
        return 503, {"ok": False, "outcome": "refused", "reason": "cancel_intent_projection_corrupt", **base}
    except Exception:
        log.warning("Host Service cancel-intent write failed for %s", task_id, exc_info=True)
        return 503, {"ok": False, "outcome": "refused", "reason": "cancel_intent_write_failed", **base}
    settled = _run_cascade_cancel(task_id)
    status = _status()
    if not settled or status not in SETTLED_STATUSES:
        return 503, {"ok": False, "outcome": "unresolved", "reason": "cancellation_did_not_settle", **base, "status": status}
    return 200, {
        "ok": True,
        "outcome": "cancelled" if status == STATUS_CANCELLED else "already_terminal",
        **base,
        "status": status,
    }


async def _api_chat_operation(request: Request) -> JSONResponse:
    """Read ONE accepted message this skill injected (#667).

    Scoped by source provenance: the canonical inbound row must carry this
    skill's ``source``; anything else is not found. No task id is accepted from
    the caller, so this stays a callback boundary, not a general task API.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:operations"):
        return _json_error("rate limit exceeded", 429)
    try:
        chat_id, client_message_id = _parse_operation_ref(request.path_params.get("operation_ref"))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    try:
        state = await asyncio.to_thread(_owned_operation_state, ctx, skill_name, chat_id, client_message_id)
    except Exception as exc:
        log.debug("Host service operation lookup failed", exc_info=True)
        return _json_error(str(exc), 500)
    if state is None:
        return _json_error("operation not found", 404)
    return JSONResponse({"ok": True, **state})


async def _api_chat_cancel(request: Request) -> JSONResponse:
    """Cancel the host work ONE accepted message of this skill started (#667).

    The cancellation owner does the work — durable intent first
    (``cancel_intents.request_cancel``), then custody through the cascade path
    the browser Stop uses — and the answer is its typed outcome: ``cancelled``,
    ``already_terminal``, ``unresolved`` (custody did not confirm settlement,
    proving neither live nor ended work) or ``cancel_unsupported`` (nothing this
    request started is addressable yet: still queued, or a message delivered
    into a pre-existing task). Never a ``cancelled`` that did not happen.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _token_payload = await _authenticated(ctx, request.headers.get("x-skill-token", ""), "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:cancel"):
        return _json_error("rate limit exceeded", 429)
    try:
        body = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    if not isinstance(body, dict):
        return _json_error("request body must be a JSON object", 400)
    try:
        chat_id, client_message_id = _parse_operation_ref(body.get("operation_ref"))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    reason = " ".join(str(body.get("reason") or "").split())[:500]
    try:
        status, payload = await asyncio.to_thread(
            _cancel_owned_operation, ctx, skill_name, chat_id, client_message_id, reason,
        )
    except Exception as exc:
        log.warning("Host service operation cancel failed", exc_info=True)
        return _json_error(str(exc), 503)
    return JSONResponse(payload, status_code=status)


async def _ws_events(websocket: WebSocket) -> None:
    ctx: HostServiceContext = websocket.app.state.host_service_context
    try:
        skill_name, token_payload = await _authenticated(ctx, _token_from_websocket(websocket))
    except HostServiceAuthError:
        await websocket.close(code=1008)
        return
    offered = set(websocket.scope.get("subprotocols") or [])
    selected_protocol = "ouroboros.host.events.v1" if "ouroboros.host.events.v1" in offered else None
    await websocket.accept(subprotocol=selected_protocol)
    subscriptions: list[str] = []
    loop = asyncio.get_running_loop()
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "skill": skill_name})
            elif message.get("type") == "subscribe":
                topic = str(message.get("topic") or "")
                try:
                    await asyncio.to_thread(
                        ctx.require_permission, skill_name, token_payload, f"subscribe_event:{topic}",
                    )
                except HostServiceAuthError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    continue

                subscribed_topic = topic

                def _send_event(payload: Dict[str, Any], event_topic: str = subscribed_topic) -> None:
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"type": "event", "topic": event_topic, "data": payload}),
                        loop,
                    )

                sub_id = get_global_event_bus().subscribe(skill_name, topic, _send_event)
                subscriptions.append(sub_id)
                await websocket.send_json({"type": "subscribed", "topic": topic})
            else:
                await websocket.send_json({"type": "error", "error": "unsupported message type"})
    except WebSocketDisconnect:
        return
    finally:
        bus = get_global_event_bus()
        for sub_id in subscriptions:
            bus.unsubscribe(sub_id)


def create_host_service_app(
    data_dir: pathlib.Path,
    *,
    bridge_getter: Optional[Callable[[], Any]] = None,
    tool_schemas_getter: Optional[Callable[[], list[dict[str, Any]]]] = None,
    ws_broadcaster_getter: Optional[Callable[[], Callable[[dict], None]]] = None,
    presence_runner: Optional[Callable[..., Any]] = None,
) -> Starlette:
    # host_notify imports this module's helpers; mounted here, imported lazily.
    from ouroboros.gateway.host_notify import _api_notify
    from ouroboros.gateway.host_call import host_call_routes

    app = Starlette(
        routes=[
            *host_call_routes(),
            Route("/identity", _api_identity, methods=["GET"]),
            Route("/tools/schemas", _api_tool_schemas, methods=["GET"]),
            Route("/chat/allocate-internal", _api_allocate_internal, methods=["POST"]),
            Route("/chat/inject", _api_chat_inject, methods=["POST"]),
            Route("/chat/operations/{operation_ref:path}", _api_chat_operation, methods=["GET"]),
            Route("/chat/cancel", _api_chat_cancel, methods=["POST"]),
            Route("/chat/decision", _api_chat_decision, methods=["POST"]),
            Route("/ui/language", _api_ui_language, methods=["POST"]),
            Route("/presence/place", _api_presence_place, methods=["GET"]),
            Route("/presence/turn", _api_presence_turn, methods=["POST"]),
            Route("/presence/delivery", _api_presence_delivery, methods=["POST"]),
            Route("/presence/work/{work_ref}", _api_presence_work, methods=["GET", "POST"]),
            Route("/ui/ws-message", _api_ws_message, methods=["POST"]),
            Route("/notify", _api_notify, methods=["POST"]),
            WebSocketRoute("/events", _ws_events),
        ]
    )
    app.state.host_service_context = HostServiceContext(
        pathlib.Path(data_dir),
        bridge_getter=bridge_getter,
        tool_schemas_getter=tool_schemas_getter,
        ws_broadcaster_getter=ws_broadcaster_getter,
        presence_runner=presence_runner,
    )
    return app


def host_service_port() -> int:
    return int(os.environ.get("OUROBOROS_HOST_SERVICE_PORT", str(DEFAULT_HOST_SERVICE_PORT)))


__all__ = [
    "AUTH_TOKEN_FILENAME",
    "DEFAULT_HOST_SERVICE_HOST",
    "DEFAULT_HOST_SERVICE_PORT",
    "HostServiceContext",
    "create_host_service_app",
    "host_service_port",
]
