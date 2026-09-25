"""Loopback-only Host Service API for privileged skill callbacks."""

from __future__ import annotations

import asyncio
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

from ouroboros.contracts.chat_id_policy import A2A_CHAT_ID_MAX, A2A_CHAT_ID_MIN, WEB_UI_CHAT_ID, is_a2a_chat_id
from ouroboros.event_bus import OWNER_NOTIFICATION_TEXT_CHARS, emit_owner_notification, get_global_event_bus
from ouroboros.config import WS_RELAY_BURST, WS_RELAY_REFILL_PER_SEC
from ouroboros.gateway._helpers import run_sync_to_completion
from ouroboros.gateway.files import store_chat_upload
from ouroboros.presence_delivery import (
    DELIVERY_VERSION, PresenceDeliveryConflict, PresenceDeliveryRecorder,
    delivery_reporting_version,
)
from ouroboros.skill_loader import (
    find_skill,
    grant_status_for_skill,
    load_enabled,
)
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)
_json_error = lambda message, status=500: JSONResponse({"ok": False, "error": message}, status_code=status)

DEFAULT_HOST_SERVICE_HOST = "127.0.0.1"
DEFAULT_HOST_SERVICE_PORT = 8767
AUTH_TOKEN_FILENAME = "auth_token.json"
# ``/identity`` advertises the owner-notification contract (``POST /notify``) so a
# skill can degrade on an older host without a side-effecting probe.
NOTIFY_VERSION = 1

# The out-of-process WS progress relay (``POST /ui/ws-message``) is the one
# token-bucket lane: a 60-message burst reserve that refills one message per
# second (owner decision 2026-09-06: a burst must not silence a widget for the
# rest of a minute). Every other Host Service lane keeps the sliding window.


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
        self._counter_lock = threading.Lock()
        self.presence_deliveries = PresenceDeliveryRecorder(self.data_dir)

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

    @property
    def skills_state_dir(self) -> pathlib.Path:
        return self.data_dir / "state" / "skills"

    def authenticate_token(self, raw_token: str) -> str:
        return self.authenticate_token_payload(raw_token)[0]

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
            raise HostServiceAuthError(f"skill {skill_name!r} does not have a fresh executable review")
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
            # Not conversation content, so no owner grant: lifecycle facts and the
            # host's own owner notifications (a transport mirrors them).
            if topic in ("skill.lifecycle", "owner.notification") and "subscribe_event" in permissions and topic in declared:
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


async def _api_identity(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        ctx.authenticate_token(request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
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
    return JSONResponse({"ok": True, "name": name, "description": description,
                         "presence_delivery_version": DELIVERY_VERSION,
                         "notify_version": NOTIFY_VERSION})


async def _api_tool_schemas(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name = ctx.authenticate_token(request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:tools"):
        return _json_error("rate limit exceeded", 429)
    schemas = ctx.tool_schemas_getter()
    return JSONResponse({"ok": True, "tools": schemas})


async def _api_allocate_internal(request: Request) -> JSONResponse:
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    try:
        ctx.require_permission(skill_name, token_payload, "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    try:
        payload = await request.json()
        chat_id = ctx.allocate_internal_chat_id(skill_name, str(payload.get("range_name") or "a2a"))
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
        skill_name, token_payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    try:
        ctx.require_permission(skill_name, token_payload, "inject_chat")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:inject"):
        return _json_error("rate limit exceeded", 429)
    if not ctx._enter_inflight(skill_name):
        return _json_error("too many in-flight inject requests", 429)
    subscription_id = ""
    pending_uploads = ExitStack()
    try:
        payload = await request.json()
        text = str(payload.get("text") or "")
        image_caption = str(payload.get("image_caption") or "")
        client_message_id = str(payload.get("client_message_id") or "").strip()[:128]
        chat_id = int(payload.get("chat_id") or 0)
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
                from ouroboros.project_dialogue import _text_sha256
                from ouroboros.task_status import SETTLED_STATUSES

                if str(inbound.get("source") or "") != f"skill:{skill_name}":
                    return _json_error("client_message_id is already bound to another source", 409)
                logged = text.strip() or image_caption.strip() or (
                    "(image attached)" if str(payload.get("image_base64") or "").strip()
                    else "(file attached)" if payload.get("attachments") else ""
                )
                if _text_sha256(inbound.get("text")) != _text_sha256(logged):
                    return _json_error("client_message_id was already used for a different message", 409)
                state = _operation_state(ctx, rows, inbound)
                if state["status"] in SETTLED_STATUSES:
                    return JSONResponse({
                        "ok": True, "response": str(state.get("text") or ""),
                        "status": state["status"], "rejoined": True, **correlated,
                    })
                if not wait_for_response:
                    return JSONResponse({"ok": True, "status": "accepted", "rejoined": True, **correlated}, status_code=202)
                rejoined = True
        uploads: list[dict[str, str]] = []
        if not rejoined:
            try:
                uploads = await run_sync_to_completion(
                    _inject_attachment_uploads, ctx, skill_name, payload.get("attachments"), pending_uploads,
                )
            except ValueError as exc:
                return _json_error(str(exc), 400)
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
                image_base64=str(payload.get("image_base64") or ""),
                image_mime=str(payload.get("image_mime") or ""),
                image_caption=image_caption,
                transport=payload.get("transport") if isinstance(payload.get("transport"), dict) else {},
                **({"task_metadata": {"chat_attachment_uploads": uploads}} if uploads else {}),
                **({"client_message_id": client_message_id} if client_message_id else {}),
            )
            if client_message_id:
                from supervisor.message_bus import accept_local_message

                try:
                    _, rejoined = await run_sync_to_completion(
                        accept_local_message, bridge, ctx.data_dir, text,
                        retain_inputs=pending_uploads.pop_all, **message,
                    )
                except ValueError as exc:
                    return _json_error(str(exc), 409)
            else:
                bridge.enqueue_local_message(text, **message)
                pending_uploads.pop_all()
        if not wait_for_response:
            if rejoined:
                return JSONResponse({"ok": True, "status": "accepted", "rejoined": True, **correlated}, status_code=202)
            return JSONResponse({"ok": True, "status": "queued", **correlated}, status_code=202)
        deadline = time.monotonic() + timeout
        while not response_event.is_set():
            if client_message_id:
                from ouroboros.task_status import SETTLED_STATUSES

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


def _inject_attachment_uploads(
    ctx: HostServiceContext, skill_name: str, value: Any, cleanup: ExitStack,
) -> list[dict[str, str]]:
    """Copy a skill's inbound files into the shared chat-upload store (#668).

    Each ``{path, name?, mime?}`` must be a regular file under the calling
    skill's OWN state root (the ``staged_files`` confinement; a symlink that
    resolves outside is refused). The host copies it through the SAME store the
    browser paperclip uses — ``data/uploads``, unique name, verified bytes — so the
    worker's ``stage_task_attachments`` and the secret-name rule see one upload
    family. Returns ``chat_attachment_uploads`` specs (``{path, label, mime}``);
    the skill removes its parked copy afterwards. Each new destination belongs
    to the request cleanup stack until its message is accepted; partial batches
    and cancelled copy waits therefore cannot orphan successful earlier copies.
    """
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise ValueError("attachments must be a list of {path, name?, mime?}")
    if len(value) > _INJECT_ATTACHMENT_MAX:
        raise ValueError(f"attachments: at most {_INJECT_ATTACHMENT_MAX} files per message")
    state_root = (ctx.skills_state_dir / skill_name).resolve(strict=False)
    specs: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"attachments[{index}] must be an object")
        source = pathlib.Path(str(item.get("path") or "")).expanduser().resolve(strict=False)
        try:
            source.relative_to(state_root)
        except ValueError as exc:
            raise ValueError(f"attachments[{index}] is outside this skill's state") from exc
        if not source.is_file():
            raise ValueError(f"attachments[{index}] is not a regular file")
        name = os.path.basename(str(item.get("name") or "").strip()) or source.name
        stored = store_chat_upload(source, name, data_dir=ctx.data_dir)
        cleanup.callback(stored.unlink, missing_ok=True)
        specs.append({"path": str(stored), "label": name, "mime": str(item.get("mime") or "")})
    return specs


def _presence_staged_files(
    ctx: HostServiceContext,
    skill_name: str,
    value: Any,
) -> tuple[pathlib.Path, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError("staged_files must be a list of paths")
    state_root = (ctx.skills_state_dir / skill_name).resolve(strict=False)
    files = []
    for index, raw in enumerate(value):
        # Keep the host boundary responsible only for request shape and source
        # confinement.  Missing/non-file inputs and the staging limit belong to
        # the existing canonical staging owner, which emits the complete typed
        # ordinal manifest before Presence can call the model.
        path = pathlib.Path(str(raw or "")).expanduser().resolve(strict=False)
        try:
            path.relative_to(state_root)
        except ValueError as exc:
            raise ValueError(f"staged_files[{index}] is outside this skill's state") from exc
        files.append(path)
    return tuple(files)


def _owner_notification_chat_id() -> int:
    """The owner's canonical chat, or Main while no owner is bound.

    Deliberately not ``notification_chat_route``: that helper keeps chat 0 as a
    real destination (the Skill Review panel), and a notification addressed
    there would be refused by the browser notifier's room gate and read by
    nobody.
    """
    from supervisor.state import load_state

    try:
        owner = int(load_state().get("owner_chat_id") or 0)
    except (TypeError, ValueError):
        owner = 0
    return owner if owner > 0 else WEB_UI_CHAT_ID


async def _api_notify(request: Request) -> JSONResponse:
    """One owner notification from a skill: never a chat row, never a model turn.

    The host, not the skill, stamps the source; the fact is one durable
    ``owner_notification`` events row (its live browser frame is that append's
    log-sink copy) plus the ``owner.notification`` topic for transport skills.
    ``key`` is the producer's own identity for the notice, so a redelivery
    after a lost acknowledgement collapses on the client instead of ringing
    twice. A failed durable write is 503 — the skill retries the same notice.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(
            request.headers.get("x-skill-token", "")
        )
        ctx.require_permission(skill_name, token_payload, "notify_owner")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:notify"):
        return _json_error("rate limit exceeded", 429)
    try:
        payload = await request.json()
    except Exception:
        return _json_error("invalid json", 400)
    if not isinstance(payload, dict):
        return _json_error("request body must be a JSON object", 400)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return _json_error("text is required", 400)
    if len(text.strip()) > OWNER_NOTIFICATION_TEXT_CHARS:
        return _json_error(f"text must be at most {OWNER_NOTIFICATION_TEXT_CHARS} characters", 400)
    key = payload.get("key", "")
    if key is not None and not isinstance(key, str):
        return _json_error("key must be a string", 400)
    key = (key or "").strip()
    if len(key) > 128:
        return _json_error("key must be at most 128 characters", 400)
    at_raw = payload.get("at")
    cron_raw = payload.get("cron")
    cancel = payload.get("cancel", False)
    if cancel not in (True, False):
        return _json_error("cancel must be a boolean", 400)
    if at_raw is not None or cron_raw is not None or cancel:
        return await run_sync_to_completion(
            _schedule_owner_notification, ctx, skill_name, text.strip(), key,
            at_raw, cron_raw, payload.get("timezone"), bool(cancel),
        )
    try:
        # The append takes the log's file lock: off the event loop, like every
        # other durable write this service performs.
        row = await run_sync_to_completion(
            emit_owner_notification, ctx.data_dir, chat_id=_owner_notification_chat_id(),
            category="notice", text=text, source=f"skill:{skill_name}", key=key,
        )
    except ValueError as exc:
        return _json_error(str(exc), 400)
    if row is None:
        return _json_error("notification log write failed; retry the same notice", 503)
    return JSONResponse({"ok": True, "ts": row["ts"], "chat_id": row["chat_id"]})


def _notify_schedule_id(skill_name: str, key: str) -> str:
    """One row per (skill, key): the slug for humans, a hash so two keys that
    slug alike (`встреча 1` / `встреча-1`) never replace each other."""
    from hashlib import sha256

    from ouroboros.schedule_contract import schedule_slug

    return f"{schedule_slug('notify', skill_name, key)}-{sha256(key.encode('utf-8')).hexdigest()[:8]}"


def _schedule_owner_notification(ctx: "HostServiceContext", skill_name: str, text: str, key: str,
                                 at_raw: Any, cron_raw: Any, timezone_raw: Any, cancel: bool) -> JSONResponse:
    """A deferred notice is a ``kind: "notify"`` row of the ONE schedule table.

    The scheduler tick fires it without a model turn; ``key`` is the row's
    identity, so the same key re-posted moves the reminder and ``cancel`` with
    that key removes it through the audited delete. Without a key a row is
    fire-and-forget (a fresh id each time).
    """
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.schedule_contract import cron_error, timezone_error
    from supervisor.queue import (
        ScheduleRefused, ScheduleStoreUnreadable, load_schedule_store, remove_scheduled_task,
        schedule_transaction, upsert_scheduled_task,
    )

    source = f"skill:{skill_name}"
    if cancel:
        if at_raw is not None or cron_raw is not None:
            return _json_error("cancel cannot be combined with at or cron", 400)
        if not key:
            return _json_error("cancel requires the key of the scheduled notification", 400)
        schedule_id = _notify_schedule_id(skill_name, key)
        try:
            with schedule_transaction(ctx.data_dir):
                rows = load_schedule_store(ctx.data_dir).get("tasks") or []
                current = next((row for row in rows if str(row.get("id") or "") == schedule_id), None)
                if current is None or str(current.get("source") or "") != source:
                    return _json_error("no scheduled notification with that key", 404)
                changed = remove_scheduled_task(
                    schedule_id, drive_root=ctx.data_dir, actor=source,
                    reason="notification cancelled by its skill")
        except ScheduleStoreUnreadable as exc:
            return _json_error(str(exc), 503)
        return JSONResponse({"ok": True, "cancelled": bool(changed), "id": schedule_id})
    if (at_raw is None) == (cron_raw is None):
        return _json_error("supply exactly one of at (ISO 8601 instant) or cron (5-field expression)", 400)
    timezone = str(timezone_raw or "").strip()
    if at_raw is not None:
        if not isinstance(at_raw, str) or parse_deadline_ts(at_raw.strip()) is None:
            return _json_error("at must be a parseable ISO 8601 instant", 400)
        trigger = {"type": "once", "run_at": parse_deadline_ts(at_raw.strip()).isoformat()}
    else:
        if not isinstance(cron_raw, str):
            return _json_error("cron must be a 5-field expression", 400)
        if err := cron_error(cron_raw.strip()):
            return _json_error(err, 400)
        trigger = {"type": "cron", "expr": cron_raw.strip()}
    if err := timezone_error(timezone):
        return _json_error(err, 400)
    schedule_id = _notify_schedule_id(skill_name, key) if key else f"notify-{skill_name}-{utc_now_iso()[:19].replace(':', '')}-{os.urandom(3).hex()}"
    # ``name`` is audit material (schedule_mutation rows carry it); the sentence
    # is not, so the row is named by its owner and the UI reads the sentence
    # from ``notification`` itself.
    record = {
        "id": schedule_id, "name": f"Reminder from {skill_name}", "kind": "notify", "source": source,
        "enabled": True, "timezone": timezone, "trigger": trigger,
        "notification": {"text": text, "key": key},
    }
    try:
        with schedule_transaction(ctx.data_dir):
            rows = load_schedule_store(ctx.data_dir).get("tasks") or []
            current = next((row for row in rows if str(row.get("id") or "") == schedule_id), None)
            if current is not None:
                if str(current.get("source") or "") != source:
                    return _json_error("that schedule id belongs to another source", 409)
                if str(current.get("manual_override") or "") in ("disabled", "deleted"):
                    # The owner switched this reminder off; a skill's repeat of the
                    # same key does not lift that — only the owner's restore does.
                    return JSONResponse({"ok": True, "scheduled": False, "id": schedule_id,
                                         "status": "suppressed"})
            stored = upsert_scheduled_task(
                record, drive_root=ctx.data_dir, actor=source,
                reason="notification scheduled by its skill")
    except ScheduleRefused as refusal:
        return _json_error(refusal.message, 409 if refusal.status == "audit_unavailable" else 400)
    except ScheduleStoreUnreadable as exc:
        return _json_error(str(exc), 503)
    return JSONResponse({
        "ok": stored.get("audit") == "recorded", "scheduled": True, "id": schedule_id,
        "next_run_at": stored.get("next_run_at") or trigger.get("run_at") or "",
    })


async def _api_presence_delivery(request: Request) -> JSONResponse:
    """Record exact provider receipts without sending or starting model work."""
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(
            request.headers.get("x-skill-token", "")
        )
        ctx.require_permission(skill_name, token_payload, "presence")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:presence_delivery"):
        return _json_error("rate limit exceeded", 429)
    # Receipts, turns and inject have separate in-flight budgets: five long turns must
    # not starve the receipts those turns' own sends produce.
    if not ctx._enter_inflight(f"{skill_name}:delivery"):
        return _json_error("too many in-flight presence requests", 429)
    try:
        payload = await request.json()
        result = await run_sync_to_completion(ctx.presence_deliveries.record, skill_name, payload)
        return JSONResponse(result)
    except PresenceDeliveryConflict as exc:
        return _json_error(str(exc), 409)
    except (ValueError, TypeError) as exc:
        return _json_error(str(exc), 400)
    except Exception:
        log.warning("Presence delivery history write failed for skill %s", skill_name, exc_info=True)
        return _json_error("presence delivery history write failed; retry the same receipt", 503)
    finally:
        ctx._leave_inflight(f"{skill_name}:delivery")


async def _api_presence_turn(request: Request) -> JSONResponse:
    """Run one non-owner event under a host-resolved reviewed profile ceiling."""

    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(
            request.headers.get("x-skill-token", "")
        )
        ctx.require_permission(skill_name, token_payload, "presence")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    if not ctx.rate_limiter.allow(f"{skill_name}:presence"):
        return _json_error("rate limit exceeded", 429)
    from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import PresenceTurnError, PresenceTurnEvent

    if not ctx._enter_inflight(f"{skill_name}:presence"):
        return _json_error("too many in-flight presence requests", 429)
    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) - {
            "binding_id", "event", "staged_files", "delivery_reporting_version",
        }:
            return _json_error("invalid presence payload", 400)
        reporting_version = delivery_reporting_version(payload.get("delivery_reporting_version", 0))
        event_payload = payload.get("event")
        expected = {
            "source_event_id", "provider", "account_id", "conversation_id", "thread_id",
            "conversation_key", "actor", "conversation", "message", "text",
        }
        if not isinstance(event_payload, dict) or set(event_payload) != expected:
            return _json_error("invalid presence event", 400)

        from ouroboros.loop import _resolve_loop_max_rounds
        admission = admit_presence_turn(
            drive_root=ctx.data_dir,
            authenticated_transport_skill=skill_name,
            binding_id=str(payload.get("binding_id") or ""),
            global_max_rounds=_resolve_loop_max_rounds(),
        )
        provider = str(event_payload.get("provider") or "").strip()
        account_id = str(event_payload.get("account_id") or "").strip()
        conversation_id = str(event_payload.get("conversation_id") or "").strip()
        thread_id = str(event_payload.get("thread_id") or "").strip()
        if (
            provider != admission.origin.transport
            or account_id != admission.origin.account_id
            or (
                admission.origin.conversation_id != "*"
                and conversation_id != admission.origin.conversation_id
            )
            or (admission.origin.thread_id and thread_id != admission.origin.thread_id)
        ):
            return _json_error("presence event does not match its owner-created binding", 403)
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
            actor=dict(event_payload["actor"]) if isinstance(event_payload["actor"], dict) else {},
            conversation=(
                dict(event_payload["conversation"])
                if isinstance(event_payload["conversation"], dict)
                else {}
            ),
            message=dict(event_payload["message"]) if isinstance(event_payload["message"], dict) else {},
            text=str(event_payload["text"] or ""),
            delivery_reporting_version=reporting_version,
        )
        if not event.source_event_id or not event.conversation_key or not event.actor:
            return _json_error("presence event is missing identity facts", 400)
        result = await asyncio.to_thread(
            ctx.presence_runner,
            admission=admission,
            event=event,
            staged_files=_presence_staged_files(ctx, skill_name, payload.get("staged_files")),
        )
        return JSONResponse({
            "ok": True,
            "status": "completed",
            "outcome": result.outcome,
            "text": result.text,
            "turn_ref": result.task_id,
            "work_ref": result.work_ref,
            "delivery_reporting_version": getattr(result, "delivery_reporting_version", 0),
        })
    except json.JSONDecodeError:
        return _json_error("invalid json", 400)
    except (PresenceAdmissionError, PresenceTurnError) as exc:
        payload = {"ok": False, "error": str(exc), "code": exc.code, "field": exc.field}
        attachment_manifest = getattr(exc, "attachment_manifest", None)
        if isinstance(attachment_manifest, list):
            payload["attachment_manifest"] = [
                dict(row) for row in attachment_manifest if isinstance(row, dict)
            ]
        return JSONResponse(payload, status_code=409)
    except (OSError, ValueError) as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        log.debug("Host service presence turn failed", exc_info=True)
        return _json_error(str(exc), 500)
    finally:
        ctx._leave_inflight(f"{skill_name}:presence")


async def _api_presence_work(request: Request) -> JSONResponse:
    """Return a correlated late result without exposing the general task API."""

    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(
            request.headers.get("x-skill-token", "")
        )
        ctx.require_permission(skill_name, token_payload, "presence")
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    work_ref = str(request.path_params.get("work_ref") or "").strip()
    binding_id = str(request.query_params.get("binding_id") or "").strip()
    try:
        from ouroboros.presence_bindings import load_presence_binding
        from ouroboros.task_results import load_task_result

        load_presence_binding(ctx.data_dir, skill_name, binding_id)
        stored = load_task_result(ctx.data_dir, work_ref) or {}
        metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
        presence = metadata.get("presence") if isinstance(metadata.get("presence"), dict) else {}
        if str(presence.get("binding_id") or "") != binding_id:
            return _json_error("presence work reference not found", 404)
        status = str(stored.get("status") or "")
        if status not in {"completed", "failed", "cancelled"}:
            return JSONResponse({"ok": True, "status": "pending", "work_ref": work_ref,
                                 "delivery_reporting_version": presence.get("delivery_reporting_version", 0)}, status_code=202)
        from ouroboros.presence_runner import presence_result_from_stored

        result = presence_result_from_stored(stored, work_ref)
        return JSONResponse({
            "ok": True,
            "status": status,
            "outcome": result.outcome,
            "text": result.text,
            "work_ref": work_ref,
            "delivery_reporting_version": result.delivery_reporting_version,
        })
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code:
            return _json_error(str(exc), 404)
        log.debug("Host service presence work lookup failed", exc_info=True)
        return _json_error("presence work lookup failed", 500)


async def _api_chat_decision(request: Request) -> JSONResponse:
    """Relay the owner's answer through the shared owner-decision ingress.

    The SAME ingress as ``POST /api/decisions`` (``task_decision.answer_decision``):
    idempotent per ``request_id``, first answer wins, typed 404/409 refusals.
    ``inject_chat`` is the owner grant that already lets this skill speak as the
    owner's chat input; answering the owner's own quiz needs nothing more (#472).
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
        ctx.require_permission(skill_name, token_payload, "inject_chat")
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


async def _api_ws_message(request: Request) -> JSONResponse:
    """WS-out bridge: relay a namespaced extension WS event to browser clients.

    Identity is derived from the token (never the body); the host re-derives the
    ``ext_<len>_<token>_<short>`` namespace, so an out-of-process child/companion
    cannot spoof another skill's events. ``ws_handler`` is a manifest permission,
    not an owner grant, mirroring the in-process ``send_ws_message`` check.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, _payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
    except HostServiceAuthError as exc:
        return _json_error(str(exc), 403)
    loaded = find_skill(ctx.data_dir, skill_name)
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


# --- A2A operation correlation (#667) ---------------------------------------
#
# An accepted inbound message is identified by the (chat_id, client_message_id)
# pair the caller supplied; ``operation_ref`` is that pair spelled
# ``"<chat_id>:<client_message_id>"``, so a caller can address the operation
# before its own inject wait has returned. The host keeps NO new record: the
# canonical inbound source, actual task origin and in-flight turn registry
# are the authorities. Annotation/outbound ids only help discover candidates.


def operation_ref(chat_id: int, client_message_id: str) -> str:
    return f"{int(chat_id)}:{client_message_id}"


def _parse_operation_ref(value: Any) -> tuple[int, str]:
    head, sep, tail = str(value or "").partition(":")
    if not sep or not tail.strip():
        raise ValueError("operation_ref must be '<chat_id>:<client_message_id>'")
    return int(head), tail.strip()[:128]


def _chat_rows(ctx: HostServiceContext, chat_id: int) -> list:
    """Read retained canonical sources; a display tail cannot prove absence."""
    from ouroboros.utils import iter_jsonl_objects, jsonl_chain_handles

    with jsonl_chain_handles(ctx.data_dir / "logs" / "chat.jsonl", strict=True) as handles:
        return [row for path, handle in handles for row in iter_jsonl_objects(path, _handle=handle)
                if row.get("chat_id") == chat_id]


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
    when the host session that accepted the message is gone and nothing else
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
    targets = dict.fromkeys([*queued, str(receipt.get("target") or ""),
                            *(str(row.get("task_id") or "") for row in reversed(rows))])
    for target in targets:
        if not target:
            continue
        try:
            raw = queued.get(target) or load_task_result(ctx.data_dir, target, strict=True) or {}
        except ValueError:
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
    accepted_session = str(inbound.get("session_id") or "")
    live_session = str((read_json_dict(ctx.data_dir / "state" / "state.json") or {}).get("session_id") or "")
    if accepted_session and live_session and accepted_session != live_session:
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
        skill_name, token_payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
        ctx.require_permission(skill_name, token_payload, "inject_chat")
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
    ``already_terminal``, ``unresolved`` (custody did not settle; the work is
    still live) or ``cancel_unsupported`` (nothing this request started is
    addressable yet: still queued, or a message delivered into a pre-existing
    task). Never a ``cancelled`` that did not happen.
    """
    ctx: HostServiceContext = request.app.state.host_service_context
    try:
        skill_name, token_payload = ctx.authenticate_token_payload(request.headers.get("x-skill-token", ""))
        ctx.require_permission(skill_name, token_payload, "inject_chat")
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
        skill_name, token_payload = ctx.authenticate_token_payload(_token_from_websocket(websocket))
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
                    ctx.require_permission(skill_name, token_payload, f"subscribe_event:{topic}")
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
    app = Starlette(
        routes=[
            Route("/identity", _api_identity, methods=["GET"]),
            Route("/tools/schemas", _api_tool_schemas, methods=["GET"]),
            Route("/chat/allocate-internal", _api_allocate_internal, methods=["POST"]),
            Route("/chat/inject", _api_chat_inject, methods=["POST"]),
            Route("/chat/operations/{operation_ref:path}", _api_chat_operation, methods=["GET"]),
            Route("/chat/cancel", _api_chat_cancel, methods=["POST"]),
            Route("/chat/decision", _api_chat_decision, methods=["POST"]),
            Route("/presence/turn", _api_presence_turn, methods=["POST"]),
            Route("/presence/delivery", _api_presence_delivery, methods=["POST"]),
            Route("/presence/work/{work_ref}", _api_presence_work, methods=["GET"]),
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
