"""Loopback-only Host Service API for privileged skill callbacks."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import pathlib
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from ouroboros.contracts.chat_id_policy import A2A_CHAT_ID_MAX, A2A_CHAT_ID_MIN, is_a2a_chat_id
from ouroboros.event_bus import get_global_event_bus
from ouroboros.gateway.files import ChatUploadPayloadTooLarge, store_chat_upload
from ouroboros.skill_loader import (
    find_skill,
    grant_status_for_skill,
    load_enabled,
    review_status_allows_execution,
)
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)
_json_error = lambda message, status=500: JSONResponse({"ok": False, "error": message}, status_code=status)

DEFAULT_HOST_SERVICE_HOST = "127.0.0.1"
DEFAULT_HOST_SERVICE_PORT = 8767
AUTH_TOKEN_FILENAME = "auth_token.json"


class HostServiceAuthError(Exception):
    """Raised when a skill token cannot be authenticated."""


class _RateLimiter:
    def __init__(self, limit: int = 60, window_sec: float = 60.0):
        self.limit = limit
        self.window_sec = window_sec
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def _sweep(self, now: float) -> None:
        # Drop keys idle past the window so _hits does not grow unbounded as
        # distinct skill keys ({skill}:{endpoint}) churn over the process
        # lifetime. Must pop each key's stale timestamps FIRST, then delete the
        # ones left empty (an idle key still holds stale, un-popped entries).
        # Collect-then-delete avoids mutating the dict during iteration.
        # Caller holds self._lock.
        stale = []
        for key, hits in self._hits.items():
            while hits and now - hits[0] > self.window_sec:
                hits.popleft()
            if not hits:
                stale.append(key)
        for key in stale:
            del self._hits[key]

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            # Amortized cleanup: at most once per window, under the existing lock.
            if now - self._last_sweep > self.window_sec:
                self._sweep(now)
                self._last_sweep = now
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_sec:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


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
        self.rate_limiter = _RateLimiter()
        self._inflight: Dict[str, int] = defaultdict(int)
        self._inflight_lock = threading.Lock()
        self._counter_lock = threading.Lock()

    def _default_bridge(self) -> Any:
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is None:
            raise RuntimeError("message bridge is not initialized")
        return bridge

    def _default_tool_schemas(self) -> list[dict[str, Any]]:
        try:
            from supervisor.workers import _get_chat_agent

            return list(_get_chat_agent().tools.schemas())
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
        if not review_status_allows_execution(loaded.review.status) or loaded.review.is_stale_for(loaded.content_hash):
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
    return JSONResponse({"ok": True, "name": name, "description": description})


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
    try:
        payload = await request.json()
        text = str(payload.get("text") or "")
        image_caption = str(payload.get("image_caption") or "")
        client_message_id = str(payload.get("client_message_id") or "").strip()[:128]
        try:
            uploads = _inject_attachment_uploads(ctx, skill_name, payload.get("attachments"))
        except ChatUploadPayloadTooLarge as exc:
            return _json_error(str(exc), 413)
        except ValueError as exc:
            return _json_error(str(exc), 400)
        bridge = ctx.bridge_getter()
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
        response_event: asyncio.Event = asyncio.Event()
        response_holder: dict[str, str] = {}
        if wait_for_response:
            loop = asyncio.get_running_loop()

            def on_response(response_text: str) -> None:
                response_holder["text"] = response_text
                loop.call_soon_threadsafe(response_event.set)

            subscription_id = bridge.subscribe_response(chat_id, on_response)
        bridge.enqueue_local_message(
            text,
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
        if not wait_for_response:
            return JSONResponse({"ok": True, "status": "queued"}, status_code=202)
        timeout = max(1, min(int(payload.get("timeout_sec") or 1800), 1800))
        deadline = time.monotonic() + timeout
        while not response_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _json_error("timed out waiting for response", 504)
            try:
                if await request.is_disconnected():
                    return _json_error("client disconnected", 499)
                await asyncio.wait_for(response_event.wait(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue
        return JSONResponse({"ok": True, "response": response_holder.get("text", "")})
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
        ctx._leave_inflight(skill_name)


_INJECT_ATTACHMENT_MAX = 25


def _inject_attachment_uploads(
    ctx: HostServiceContext, skill_name: str, value: Any,
) -> list[dict[str, str]]:
    """Copy a skill's inbound files into the shared chat-upload store (#668).

    Each ``{path, name?, mime?}`` must be a regular file under the calling
    skill's OWN state root (the ``staged_files`` confinement; a symlink that
    resolves outside is refused). The host copies it through the SAME store the
    browser paperclip uses — ``data/uploads``, unique name, 50 MB cap — so the
    worker's ``stage_task_attachments`` and the secret-name rule see one upload
    family. Returns ``chat_attachment_uploads`` specs (``{path, label, mime}``);
    the skill removes its parked copy afterwards.
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
    if not ctx._enter_inflight(skill_name):
        return _json_error("too many in-flight presence requests", 429)
    from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnError, PresenceTurnEvent

    try:
        payload = await request.json()
        if not isinstance(payload, dict) or set(payload) - {"binding_id", "event", "staged_files"}:
            return _json_error("invalid presence payload", 400)
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
            conversation_key=":".join((
                provider,
                account_id,
                conversation_id,
                thread_id or "0",
            )),
            actor=dict(event_payload["actor"]) if isinstance(event_payload["actor"], dict) else {},
            conversation=(
                dict(event_payload["conversation"])
                if isinstance(event_payload["conversation"], dict)
                else {}
            ),
            message=dict(event_payload["message"]) if isinstance(event_payload["message"], dict) else {},
            text=str(event_payload["text"] or ""),
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
        ctx._leave_inflight(skill_name)


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
            return JSONResponse({"ok": True, "status": "pending", "work_ref": work_ref}, status_code=202)
        outcome = str(metadata.get("presence_outcome") or "message")
        if outcome not in {"message", "silent", "tool_delivered", "deferred"}:
            outcome = "message"
        return JSONResponse({
            "ok": True,
            "status": status,
            "outcome": outcome,
            "text": (
                str(metadata.get("presence_result_text") or stored.get("result") or "")
                if outcome in {"message", "deferred"}
                else ""
            ),
            "work_ref": work_ref,
        })
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code:
            return _json_error(str(exc), 404)
        log.debug("Host service presence work lookup failed", exc_info=True)
        return _json_error("presence work lookup failed", 500)


async def _api_chat_decision(request: Request) -> JSONResponse:
    """Relay the owner's answer to a decision card (a quiz option or free answer).

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
        status, payload = await answer_decision(ctx.data_dir, body)
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
    if not ctx.rate_limiter.allow(f"{skill_name}:ws"):
        return _json_error("rate limit exceeded", 429)
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
            Route("/chat/decision", _api_chat_decision, methods=["POST"]),
            Route("/presence/turn", _api_presence_turn, methods=["POST"]),
            Route("/presence/work/{work_ref}", _api_presence_work, methods=["GET"]),
            Route("/ui/ws-message", _api_ws_message, methods=["POST"]),
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
