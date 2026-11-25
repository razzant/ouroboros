"""Authenticated loopback client for the frozen Presence Host contract."""

from __future__ import annotations

import asyncio
import json
import math
import pathlib
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

import httpcore
import httpx


_OUTCOMES = {"message", "silent", "tool_delivered", "deferred"}
_TERMINAL_WORK_STATES = {"completed", "failed", "cancelled"}
_DISPOSITIONS = {"retained", "rejected", "blocked", "retry"}
_EARLY_CONTINUATION = 2
_BUSY_RETRY_SEC = {"conversation": 1.0, "capacity": 5.0}
PLACE_POLICY_UNREAD = "Held: It retries by itself."
SKILL_ACCESS_REFUSED = "Held: Check this skill's access in Skills."
_EVENT_KEYS = {
    "source_event_id",
    "provider",
    "account_id",
    "conversation_id",
    "thread_id",
    "conversation_key",
    "actor",
    "conversation",
    "message",
    "text",
}


class PresenceHostError(RuntimeError):
    """The local Host rejected a presence request or returned an invalid response."""


class PlaceBindingRevoked(PresenceHostError):
    """The Place binding captured when an operation was queued was disconnected or changed scope."""


class PlaceBindingUnavailable(PresenceHostError):
    """Enqueue-time Place binding could not be captured on a Places-capable Host."""

    place_binding_unavailable = True

    def __init__(self, message: str, *, code: str = "place_binding_unavailable"):
        self.code = code
        super().__init__(message)


class PresenceHostHTTPError(PresenceHostError):
    """The local Host returned a concrete HTTP rejection."""

    def __init__(self, status_code: int, detail: str, payload: Optional[Dict[str, Any]] = None):
        self.status_code = int(status_code)
        self.payload = dict(payload) if isinstance(payload, dict) else {}
        self.binding_id = ""
        super().__init__(f"presence Host returned HTTP {self.status_code}: {detail}")

    @property
    def disposition(self) -> str:
        disposition = str(self.payload.get("disposition") or "")
        if self.payload.get("ok") is not False or disposition not in _DISPOSITIONS:
            return ""
        if disposition == "retained" and not str(self.payload.get("turn_ref") or "").strip():
            return ""
        return disposition

    @property
    def busy_retry_after(self) -> Optional[float]:
        field = self.payload.get("field")
        if (
            self.disposition != "retry"
            or self.payload.get("code") != "presence_capacity_full"
            or field not in _BUSY_RETRY_SEC
        ):
            return None
        hint = self.payload.get("retry_after")
        if isinstance(hint, bool) or not isinstance(hint, (int, float)) or not math.isfinite(hint):
            hint = _BUSY_RETRY_SEC[field]
        return max(1.0, float(hint))


@dataclass(frozen=True)
class PresenceSubmission:
    status: str
    outcome: str
    text: str
    turn_ref: str
    work_ref: str
    binding_id: str
    delivery_reporting_version: int = 0
    continuation_ref: str = ""
    output_ref: str = ""
    reply_anchor: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PresenceWorkResult:
    status: str
    outcome: str = ""
    text: str = ""
    work_ref: str = ""
    delivery_reporting_version: int = 0
    code: str = ""
    child_work_ref: str = ""
    outputs: tuple[Dict[str, Any], ...] = ()
    output_ref: str = ""
    reply_anchor: Optional[Dict[str, Any]] = None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_WORK_STATES


class PresenceHostTransport(Protocol):
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]],
        timeout_sec: float,
    ) -> Dict[str, Any]: ...


_MAX_HOST_BODY_BYTES = 2 * 1024 * 1024
_MAX_HOST_ERROR_BYTES = 64 * 1024
_IDENTITY_TTL_SEC = 60.0


def _parse_places_version(body: Any) -> int:
    """Strictly validate ``/identity`` ``places_version``: ``0`` = explicit legacy, ``>=1`` = Places."""
    if not isinstance(body, dict):
        raise PresenceHostError("presence Host /identity returned a non-object response")
    version = body.get("places_version")
    if type(version) is int and version == 0:
        return 0
    if type(version) is int and version >= 1:
        return 1
    raise PresenceHostError("presence Host /identity returned missing or invalid places_version")


def _parse_place_binding_version(body: Any) -> int:
    """Detect ``presence_place_binding_version`` separately from ``places_version``."""
    if not isinstance(body, dict):
        return 0
    version = body.get("presence_place_binding_version")
    return 1 if type(version) is int and version >= 1 else 0


def _check_identity_encoding(headers: Any) -> None:
    if isinstance(headers, (list, tuple)):
        vals = [v.decode("latin-1") for k, v in headers if k.lower() == b"content-encoding"]
        enc = ",".join(vals)
    else:
        enc = str(headers.get("content-encoding") or "")
    tokens = [tok.strip().lower() for tok in enc.split(",") if tok.strip()]
    if any(tok != "identity" for tok in tokens):
        raise PresenceHostError(f"presence Host returned unexpected content-encoding {enc.strip()!r}")


def _read_sync_bounded(
    response: Any, limit: int, label: str, *, deadline: Optional[float] = None
) -> bytes:
    _check_identity_encoding(response.headers)
    buf = bytearray()
    chunks = response.iter_stream() if hasattr(response, "iter_stream") else response.iter_bytes()
    for chunk in chunks:
        if deadline is not None and time.monotonic() > deadline:
            raise PresenceHostError(f"{label} timed out")
        if len(buf) + len(chunk) > limit:
            raise PresenceHostError(f"{label} exceeds {limit} bytes")
        buf.extend(chunk)
    return bytes(buf)


class _DeadlineStream(httpcore.NetworkStream):
    """NetworkStream wrapper that enforces a single shared monotonic wall-clock deadline."""

    def __init__(self, inner: httpcore.NetworkStream, deadline: float) -> None:
        self._inner = inner
        self._deadline = deadline

    def _cap(self, timeout: Optional[float], err_cls: type[Exception] = httpcore.ReadTimeout) -> float:
        rem = self._deadline - time.monotonic()
        if rem <= 0:
            raise err_cls("Host Place binding timed out")
        return rem if timeout is None else min(rem, timeout)

    def read(self, max_bytes: int, timeout: Optional[float] = None) -> bytes:
        return self._inner.read(max_bytes, timeout=self._cap(timeout))

    def write(self, buffer: bytes, timeout: Optional[float] = None) -> None:
        self._inner.write(buffer, timeout=self._cap(timeout, httpcore.WriteTimeout))

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self, ssl_context: Any, server_hostname: Optional[str] = None, timeout: Optional[float] = None
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._inner.start_tls(ssl_context, server_hostname=server_hostname, timeout=self._cap(timeout)),
            self._deadline,
        )

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _DeadlineBackend(httpcore.SyncBackend):
    """SyncBackend whose sockets strictly enforce a shared wall-clock deadline across requests."""

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        rem = self._deadline - time.monotonic()
        if rem <= 0:
            raise httpcore.ConnectTimeout("Host Place binding timed out")
        return _DeadlineStream(
            super().connect_tcp(
                host,
                port,
                timeout=rem if timeout is None else min(rem, timeout),
                local_address=local_address,
                socket_options=socket_options,
            ),
            self._deadline,
        )


class UrllibPresenceHostTransport:
    """Async no-proxy, no-redirect loopback Host transport that aborts cleanly on task cancellation."""

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]],
        timeout_sec: float,
    ) -> Dict[str, Any]:
        request_headers = {**headers, "Accept": "application/json", "Accept-Encoding": "identity"}
        content = None
        if payload is not None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8", "replace")
            request_headers["Content-Type"] = "application/json"

        async def _execute() -> bytes:
            async with httpx.AsyncClient(
                timeout=timeout_sec,
                trust_env=False,
                proxy=None,
                follow_redirects=False,
            ) as client:
                async with client.stream(method, url, headers=request_headers, content=content) as response:
                    status = response.status_code
                    if status >= 400:
                        _check_identity_encoding(response.headers)
                        err_buf = bytearray()
                        async for chunk in response.aiter_bytes():
                            err_buf.extend(chunk[: max(0, _MAX_HOST_ERROR_BYTES - len(err_buf))])
                            if len(err_buf) >= _MAX_HOST_ERROR_BYTES:
                                break
                        err_body = bytes(err_buf)
                        raise PresenceHostHTTPError(status, _error_detail(err_body), _error_payload(err_body))
                    if 300 <= status < 400:
                        raise PresenceHostHTTPError(status, f"presence Host returned unexpected redirect HTTP {status}")
                    _check_identity_encoding(response.headers)
                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(buf) + len(chunk) > _MAX_HOST_BODY_BYTES:
                            raise PresenceHostError("presence Host response exceeds 2 MiB")
                        buf.extend(chunk)
                    return bytes(buf)

        try:
            data = await asyncio.wait_for(_execute(), timeout=timeout_sec)
        except asyncio.CancelledError:
            raise
        except PresenceHostError:
            raise
        except (httpx.HTTPError, TimeoutError, asyncio.TimeoutError, OSError) as exc:
            raise PresenceHostError(f"presence Host request failed: {type(exc).__name__}") from exc
        try:
            result = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PresenceHostError("presence Host returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise PresenceHostError("presence Host returned a non-object response")
        return result


def bind_place_sync(
    *,
    host_base: str,
    token_provider: Callable[[], str],
    account_id: str,
    conversation_id: str,
    thread_id: str = "",
    places_version: Optional[int] = None,
    place_binding_version: Optional[int] = None,
    delivery_reporting_status: str = "unknown",
    timeout_sec: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Capture an enqueue-time Place binding snapshot synchronously via loopback Host Service.

    Returns ``None`` when the Host explicitly reports ``places_version == 0`` (legacy Host), or when a
    Places-capable Host explicitly lacks ``presence_place_binding_version`` and confirms the Place is active.
    Raises :class:`PlaceBindingUnavailable` when a Places-capable Host cannot confirm or bind the Place.
    """
    discovered = delivery_reporting_status in {"supported", "unsupported"}
    if discovered and type(places_version) is int and places_version == 0:
        return None
    base = str(host_base).rstrip("/")
    try:
        token = str(token_provider() or "").strip()
        if not token or not account_id or not conversation_id:
            raise PlaceBindingUnavailable("Place binding cannot be confirmed without host credentials and target.")
        headers = {"X-Skill-Token": token, "Accept": "application/json", "Accept-Encoding": "identity"}
        deadline = time.monotonic() + max(0.05, float(timeout_sec))

        def _timeout_ext() -> Dict[str, Any]:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise PlaceBindingUnavailable("Host Place binding timed out. Nothing was queued.")
            return {"timeout": {"connect": rem, "read": rem, "write": rem, "pool": rem}}

        with httpcore.ConnectionPool(network_backend=_DeadlineBackend(deadline), http1=True, http2=False) as pool:
            binding_ver = place_binding_version if discovered and type(place_binding_version) is int else None
            if not discovered or binding_ver is None:
                with pool.stream("GET", f"{base}/identity", headers=headers, extensions=_timeout_ext()) as id_resp:
                    if id_resp.status != 200:
                        raise PlaceBindingUnavailable(
                            f"Host identity returned HTTP {id_resp.status}; nothing was queued."
                        )
                    id_raw = _read_sync_bounded(
                        id_resp, _MAX_HOST_BODY_BYTES, "presence Host /identity response", deadline=deadline
                    )
                id_body = json.loads(id_raw.decode("utf-8"))
                if _parse_places_version(id_body) == 0:
                    return None
                binding_ver = _parse_place_binding_version(id_body)
            if binding_ver == 0:
                # Host supports Places reads (places_version>=1) without place-binding snapshots:
                # verify the Place exists and is accessible without expecting a binding envelope.
                status_q = urllib.parse.urlencode({
                    "provider": "telegram",
                    "account_id": str(account_id),
                    "conversation_id": str(conversation_id),
                    **({"thread_id": str(thread_id)} if thread_id else {}),
                })
                with pool.stream("GET", f"{base}/presence/place?{status_q}", headers=headers, extensions=_timeout_ext()) as resp:
                    if resp.status >= 400:
                        raw_err = _read_sync_bounded(resp, _MAX_HOST_ERROR_BYTES, "presence Host error", deadline=deadline)
                        err_code = str(_error_payload(raw_err).get("code") or "")
                        if err_code in {"place_not_connected", "place_unattributable"}:
                            raise PlaceBindingUnavailable(
                                "This chat is not a connected Place in Circles. Nothing was queued.",
                                code="place_not_connected",
                            )
                        if err_code == "presence_auth_blocked":
                            raise PlaceBindingUnavailable(
                                "Check this skill's access in Skills. Nothing was queued.", code="presence_auth_blocked"
                            )
                        raise PlaceBindingUnavailable(
                            f"Host could not verify Place (HTTP {resp.status}). Nothing was queued."
                        )
                    if resp.status != 200:
                        raise PlaceBindingUnavailable(
                            f"Host could not verify Place (HTTP {resp.status}). Nothing was queued."
                        )
                    raw_ok = _read_sync_bounded(
                        resp, _MAX_HOST_BODY_BYTES, "presence Host place response", deadline=deadline
                    )
                status_body = json.loads(raw_ok.decode("utf-8"))
                if not isinstance(status_body, dict) or not str(status_body.get("state") or "").strip():
                    raise PlaceBindingUnavailable("Host returned an invalid Place status. Nothing was queued.")
                return None
            query = urllib.parse.urlencode({
                "operation": "bind",
                "provider": "telegram",
                "account_id": str(account_id),
                "conversation_id": str(conversation_id),
                "require_place": "true",
                **({"thread_id": str(thread_id)} if thread_id else {}),
            })
            with pool.stream("GET", f"{base}/presence/place?{query}", headers=headers, extensions=_timeout_ext()) as resp:
                status = resp.status
                if status >= 400:
                    raw_err = _read_sync_bounded(resp, _MAX_HOST_ERROR_BYTES, "presence Host error", deadline=deadline)
                    err_body = _error_payload(raw_err)
                    err_code = str(err_body.get("code") or "")
                    if err_code in {"place_not_connected", "place_unattributable"}:
                        raise PlaceBindingUnavailable(
                            "This chat is not a connected Place in Circles. Nothing was queued.",
                            code="place_not_connected",
                        )
                    if err_code == "presence_auth_blocked":
                        raise PlaceBindingUnavailable(
                            "Check this skill's access in Skills. Nothing was queued.", code="presence_auth_blocked"
                        )
                    raise PlaceBindingUnavailable(
                        f"Host could not bind Place (HTTP {status}). Nothing was queued."
                    )
                if status != 200:
                    raise PlaceBindingUnavailable(
                        f"Host could not bind Place (HTTP {status}). Nothing was queued."
                    )
                raw_ok = _read_sync_bounded(
                    resp, _MAX_HOST_BODY_BYTES, "presence Host place response", deadline=deadline
                )
            body = json.loads(raw_ok.decode("utf-8"))
    except PlaceBindingUnavailable:
        raise
    except Exception as exc:
        raise PlaceBindingUnavailable(
            f"Could not reach Host to bind Place ({type(exc).__name__}). Nothing was queued."
        ) from exc
    binding = body.get("binding") if isinstance(body, dict) else None
    if isinstance(binding, dict) and binding.get("version") == 1 and str(binding.get("place_id") or "").strip():
        return dict(binding)
    code = str(body.get("code") or "") if isinstance(body, dict) else ""
    if code in {"place_not_connected", "place_unattributable"}:
        raise PlaceBindingUnavailable(
            "This chat is not a connected Place in Circles. Nothing was queued.", code="place_not_connected"
        )
    raise PlaceBindingUnavailable(f"Host did not return a valid Place binding ({code or 'unknown'}). Nothing was queued.")


class PresenceHostClient:
    """Submit provider events and poll deferred work through the ``presence`` Host permission."""

    def __init__(
        self,
        *,
        state_dir: pathlib.Path,
        token_provider: Callable[[], str],
        host_base: str,
        transport: Optional[PresenceHostTransport] = None,
    ):
        self.state_dir = pathlib.Path(state_dir)
        self.token_provider = token_provider
        self.host_base = str(host_base).rstrip("/")
        self.transport = transport or UrllibPresenceHostTransport()
        self.delivery_reporting_status = "unknown"
        self.delivery_reporting_version = 0
        self.places_version = 0
        self.place_binding_version = 0
        self.presence_wake_signals = 0
        self.presence_identity = 0
        self.connector_identity: Dict[str, Any] = {}
        self.presence_reaction_context = 0
        self.presence_explicit_completion = 0
        self.continuation_version = 0
        self._identity_discovered_at = 0.0

    def invalidate_identity(self) -> None:
        """Force the next capability check to re-read ``/identity`` from the Host."""
        self.delivery_reporting_status = "unknown"
        self._identity_discovered_at = 0.0

    def identity_is_fresh(self) -> bool:
        """True only while cached ``/identity`` protocol flags are within the identity TTL."""
        return (
            self.delivery_reporting_status in {"supported", "unsupported"}
            and self._identity_discovered_at > 0.0
            and (time.monotonic() - self._identity_discovered_at) < _IDENTITY_TTL_SEC
        )

    async def discover_delivery_support(self, *, force: bool = False) -> int:
        if (
            not force
            and self.delivery_reporting_status in {"supported", "unsupported"}
            and (time.monotonic() - self._identity_discovered_at) < _IDENTITY_TTL_SEC
        ):
            return self.delivery_reporting_version
        try:
            response = await self.transport.request_json(
                "GET", f"{self.host_base}/identity", headers=self._headers(), payload=None, timeout_sec=10.0
            )
            places_ver = _parse_places_version(response)
            self.delivery_reporting_version = int(response.get("presence_delivery_version") == 1)
            self.places_version = places_ver
            self.place_binding_version = _parse_place_binding_version(response) if places_ver else 0
            self.presence_wake_signals = int(response.get("presence_wake_signals") == 1)
            self.presence_identity = int(response.get("presence_identity") == 1)
            self.presence_reaction_context = int(
                type(response.get("presence_reaction_context")) is int and response["presence_reaction_context"] == 1
            )
            self.presence_explicit_completion = int(
                type(response.get("presence_explicit_completion")) is int
                and response["presence_explicit_completion"] == 1
            )
            versions = response.get("presence_continuation_versions")
            self.continuation_version = (
                _EARLY_CONTINUATION
                if isinstance(versions, list) and any(type(item) is int and item == _EARLY_CONTINUATION for item in versions)
                else 0
            )
            self.delivery_reporting_status = "supported" if self.delivery_reporting_version else "unsupported"
            self._identity_discovered_at = time.monotonic()
        except (PresenceHostError, OSError, TimeoutError, asyncio.TimeoutError):
            self.delivery_reporting_status = "unavailable"
            self.delivery_reporting_version = 0
            self.places_version = 0
            self.place_binding_version = 0
            self.continuation_version = 0
            self.presence_reaction_context = 0
            self.presence_explicit_completion = 0
            self._identity_discovered_at = 0.0
        return self.delivery_reporting_version

    async def place_status(self, account_id: str, conversation_id: str, thread_id: str = "") -> Dict[str, Any]:
        """Read current Place state for ``(account_id, conversation_id, thread_id)`` without side effects."""
        await self.discover_delivery_support()
        if not self.places_version:
            return {"state": "unavailable" if self.delivery_reporting_status == "unavailable" else "legacy"}
        if not account_id or not conversation_id:
            return {"state": "unavailable"}
        try:
            query = urllib.parse.urlencode({
                "provider": "telegram",
                "account_id": account_id,
                "conversation_id": conversation_id,
                **({"thread_id": thread_id} if thread_id else {}),
            })
            place = await self.transport.request_json(
                "GET", f"{self.host_base}/presence/place?{query}", headers=self._headers(), payload=None, timeout_sec=10.0
            )
            state = str(place.get("state") or "").strip()
            if not state:
                return {"state": "unavailable"}
            return {
                "state": state,
                "place_id": str(place.get("place_id") or ""),
                "label": str(place.get("label") or "Telegram"),
            }
        except PresenceHostHTTPError as exc:
            code = str(exc.payload.get("code") or "")
            if code == "place_not_connected":
                return {"state": "not_connected", "error": str(exc.payload.get("error") or "")}
            if code == "presence_auth_blocked":
                return {"state": "auth_blocked", "error": str(exc.payload.get("error") or "")}
            if exc.status_code >= 500 or exc.status_code == 404:
                self.invalidate_identity()
            return {"state": "unavailable"}
        except (PresenceHostError, OSError, TimeoutError):
            self.invalidate_identity()
            return {"state": "unavailable"}

    async def bind_place(self, account_id: str, conversation_id: str, thread_id: str = "") -> Optional[Dict[str, Any]]:
        """Capture an enqueue-time Place binding snapshot (``place_id`` + ``connected_revision``).

        Returns ``None`` on a legacy Host (``places_version == 0``) or on a Places-capable Host that does not
        advertise ``presence_place_binding_version`` (preventing infinite bind loops on older Hosts).
        Raises :class:`PlaceBindingUnavailable` on unconnected/unattributable chats, access denials, or errors.
        """
        await self.discover_delivery_support()
        if self.delivery_reporting_status not in {"supported", "unsupported"}:
            raise PlaceBindingUnavailable("Host unavailable to confirm Place binding")
        if not self.places_version or not self.place_binding_version:
            return None
        if not account_id or not conversation_id:
            raise PlaceBindingUnavailable("Place binding requires confirmed account_id and conversation_id")
        query = urllib.parse.urlencode({
            "operation": "bind",
            "provider": "telegram",
            "account_id": account_id,
            "conversation_id": conversation_id,
            "require_place": "true",
            **({"thread_id": thread_id} if thread_id else {}),
        })
        try:
            body = await self.transport.request_json(
                "GET", f"{self.host_base}/presence/place?{query}", headers=self._headers(), payload=None, timeout_sec=10.0
            )
        except PresenceHostHTTPError as exc:
            err_code = str(exc.payload.get("code") or "")
            if err_code in {"place_not_connected", "place_unattributable"}:
                raise PlaceBindingUnavailable("Add this place in Circles.", code="place_not_connected") from exc
            if err_code == "presence_auth_blocked":
                raise PlaceBindingUnavailable(
                    "Check this skill's access in Skills.", code="presence_auth_blocked"
                ) from exc
            if exc.status_code >= 500 or exc.status_code == 404:
                self.invalidate_identity()
            raise PlaceBindingUnavailable(f"Host Place bind rejected (HTTP {exc.status_code})") from exc
        except (PresenceHostError, OSError, TimeoutError) as exc:
            self.invalidate_identity()
            raise PlaceBindingUnavailable(f"Host Place bind failed ({type(exc).__name__})") from exc
        binding = body.get("binding") if isinstance(body, dict) else None
        if isinstance(binding, dict) and binding.get("version") == 1 and str(binding.get("place_id") or "").strip():
            return dict(binding)
        if isinstance(body, dict) and str(body.get("code") or "") in {"place_not_connected", "place_unattributable"}:
            raise PlaceBindingUnavailable("Add this place in Circles.", code="place_not_connected")
        self.invalidate_identity()
        raise PlaceBindingUnavailable("Host did not return a valid Place binding")

    async def place_hold(
        self, account_id: str, conversation_id: str, thread_id: str = "", *, binding: Optional[Dict[str, Any]] = None
    ) -> str:
        if binding is not None:
            if not isinstance(binding, dict) or binding.get("version") != 1:
                raise PlaceBindingRevoked("Retained place binding is malformed")
            await self.discover_delivery_support()
            if self.delivery_reporting_status not in {"supported", "unsupported"}:
                return PLACE_POLICY_UNREAD
            if not self.places_version or not self.place_binding_version:
                return "Held: Host does not support verifying this row's saved Place binding"
            query = urllib.parse.urlencode({
                "operation": "check",
                "provider": "telegram",
                "account_id": account_id,
                "conversation_id": conversation_id,
                "require_place": "true",
                "binding": json.dumps(binding, sort_keys=True, separators=(",", ":")),
                **({"thread_id": thread_id} if thread_id else {}),
            })
            try:
                res = await self.transport.request_json(
                    "GET", f"{self.host_base}/presence/place?{query}", headers=self._headers(), payload=None, timeout_sec=10.0
                )
            except PresenceHostHTTPError as exc:
                code = str(exc.payload.get("code") or "")
                if code == "presence_auth_blocked":
                    return SKILL_ACCESS_REFUSED
                if code == "place_binding_invalid":
                    raise PlaceBindingRevoked(str(exc.payload.get("error") or "Retained place binding is invalid")) from exc
                if code in {"place_not_connected", "place_disconnected", "place_scope_changed"}:
                    raise PlaceBindingRevoked(str(exc.payload.get("error") or "Place was disconnected")) from exc
                if exc.status_code >= 500 or exc.status_code == 404:
                    self.invalidate_identity()
                return PLACE_POLICY_UNREAD
            except (PresenceHostError, OSError, TimeoutError):
                self.invalidate_identity()
                return PLACE_POLICY_UNREAD
            if res.get("allowed") is True:
                return ""
            code = str(res.get("code") or "")
            state = str(res.get("state") or "unavailable")
            label = str(res.get("label") or "Telegram")
            if code in {"place_disconnected", "place_scope_changed", "place_not_connected"}:
                raise PlaceBindingRevoked(
                    f"Cancelled: {label} was disconnected or changed scope since this operation was queued"
                )
            if state in {"silent", "paused", "observe"}:
                return f"Held: {label} is {state}"
            if code == "place_unattributable":
                return f"Held: {label} cannot be attributed right now"
            if "allowed" not in res:
                self.invalidate_identity()
            return PLACE_POLICY_UNREAD
        info = await self.place_status(account_id, conversation_id, thread_id)
        state = info.get("state", "unavailable")
        if state in {"active", "legacy"}:
            return ""
        if state == "auth_blocked":
            return SKILL_ACCESS_REFUSED
        if state == "not_connected":
            words = str(info.get("error") or "")
            return f"Held: {words}" if words else "Held: Add this place in Circles."
        if state == "unavailable":
            return PLACE_POLICY_UNREAD
        return f"Held: {info.get('label') or 'Telegram'} is {state}"

    async def report_delivery(self, payload: Dict[str, Any]) -> None:
        response = await self.transport.request_json(
            "POST", f"{self.host_base}/presence/delivery", headers=self._headers(), payload=payload, timeout_sec=10.0
        )
        if response.get("ok") is not True or response.get("recorded") is not True:
            raise PresenceHostError("Presence delivery report was not acknowledged")

    async def submit(self, event: Dict[str, Any], staged_files: Sequence[pathlib.Path]) -> PresenceSubmission:
        if not isinstance(event, dict):
            raise PresenceHostError("presence event does not match the frozen contract")
        event = {k: v for k, v in event.items() if k != "_place_binding"}
        if set(event) - {"facts"} != _EVENT_KEYS:
            raise PresenceHostError("presence event does not match the frozen contract")
        mode = await self.discover_delivery_support()
        binding_id = "" if self.places_version else load_binding_id(self.state_dir)
        event = dict(event)
        if not self.places_version:
            event.pop("facts", None)
        elif not self.presence_wake_signals and "facts" in event:
            event["facts"] = [fact for fact in event["facts"] if fact != "signal.words"]
        if self.presence_identity and self.connector_identity:
            event["identity"] = self.connector_identity
        early = self.continuation_version == _EARLY_CONTINUATION
        try:
            response = await self.transport.request_json(
                "POST",
                f"{self.host_base}/presence/turn",
                headers=self._headers(),
                payload={
                    **({"binding_id": binding_id} if not self.places_version else {}),
                    "event": dict(event),
                    "staged_files": [str(path) for path in staged_files],
                    **({"delivery_reporting_version": 1} if mode else {}),
                    **({"continuation_version": _EARLY_CONTINUATION} if early else {}),
                },
                timeout_sec=1800.0,
            )
        except PresenceHostHTTPError as exc:
            exc.binding_id = binding_id
            raise
        status = str(response.get("status") or "")
        turn_ref = _utf8_ref(response.get("turn_ref"), "turn_ref")
        continuation_ref = ""
        if status == "continuing" and early:
            continuation_ref = _utf8_ref(response.get("continuation_ref"), "continuation_ref")
            if (
                response.get("continuation_version") != _EARLY_CONTINUATION
                or not continuation_ref
                or continuation_ref != turn_ref
            ):
                raise PresenceHostError("continuing presence answer lacks its version-2 continuation_ref")
        elif status != "completed":
            raise PresenceHostError("presence Host did not complete the turn request")
        outcome = str(response.get("outcome") or "")
        if outcome not in _OUTCOMES:
            raise PresenceHostError("presence Host returned an invalid outcome")
        work_ref = _utf8_ref(response.get("work_ref"), "work_ref")
        output_ref = _utf8_ref(response.get("output_ref"), "output_ref")
        if outcome == "deferred" and not work_ref and not continuation_ref:
            raise PresenceHostError("deferred presence result is missing work_ref")
        return PresenceSubmission(
            status=status,
            outcome=outcome,
            text=str(response.get("text") or ""),
            turn_ref=turn_ref,
            work_ref=work_ref,
            binding_id=binding_id,
            delivery_reporting_version=int(response.get("delivery_reporting_version") == 1),
            continuation_ref=continuation_ref,
            output_ref=output_ref,
            reply_anchor=_reply_anchor(response.get("reply_anchor")),
        )

    async def poll(self, work_ref: str, binding_id: str) -> PresenceWorkResult:
        encoded_ref = urllib.parse.quote(str(work_ref), safe="")
        query = urllib.parse.urlencode({"binding_id": str(binding_id)}) if binding_id else ""
        url = f"{self.host_base}/presence/work/{encoded_ref}" + (f"?{query}" if query else "")
        response = await self.transport.request_json(
            "GET",
            url,
            headers=self._headers(),
            payload=None,
            timeout_sec=35.0,
        )
        status = str(response.get("status") or "")
        outputs = _selected_outputs(response)
        child = _child_ref(response, str(work_ref))
        if status == "pending":
            return PresenceWorkResult(status=status, work_ref=str(work_ref), child_work_ref=child, outputs=outputs)
        if status == "unresolved":
            return PresenceWorkResult(
                status=status,
                work_ref=str(work_ref),
                code=str(response.get("code") or ""),
                child_work_ref=child,
                outputs=outputs,
            )
        if status not in _TERMINAL_WORK_STATES:
            raise PresenceHostError("presence Host returned an invalid work status")
        outcome = str(response.get("outcome") or "")
        if outcome not in _OUTCOMES:
            raise PresenceHostError("presence Host returned an invalid work outcome")
        returned_ref = str(response.get("work_ref") or "").strip()
        if returned_ref and returned_ref != str(work_ref):
            raise PresenceHostError("presence Host returned a different work_ref")
        return PresenceWorkResult(
            status=status,
            outcome=outcome,
            text=str(response.get("text") or ""),
            work_ref=str(work_ref),
            delivery_reporting_version=int(response.get("delivery_reporting_version") == 1),
            child_work_ref=child,
            outputs=outputs,
            output_ref=str(response.get("output_ref") or ""),
            reply_anchor=_reply_anchor(response.get("reply_anchor")),
        )

    def _headers(self) -> Dict[str, str]:
        token = str(self.token_provider() or "").strip()
        if not token:
            raise PresenceHostError("presence skill token is unavailable")
        return {"X-Skill-Token": token}


def load_binding_id(state_dir: pathlib.Path) -> str:
    """Read an opaque legacy binding from skill-local settings on pre-Places hosts."""
    path = pathlib.Path(state_dir) / "settings.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PresenceHostError("presence binding is not configured") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PresenceHostError("presence binding settings are unreadable") from exc
    binding_id = payload.get("binding_id") if isinstance(payload, dict) else None
    if not isinstance(binding_id, str) or not binding_id.strip():
        raise PresenceHostError("presence binding is not configured")
    return binding_id.strip()


def _selected_outputs(response: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
    outputs = list(response.get("outputs") or [])
    if response.get("output_ref") and response.get("text"):
        outputs.append({key: response.get(key) for key in ("output_ref", "outcome", "text", "reply_anchor")})
    selected = []
    for output in outputs:
        if not isinstance(output, dict) or not output.get("output_ref") or output.get("outcome") not in _OUTCOMES:
            raise PresenceHostError("invalid released presence output")
        if output["outcome"] in {"message", "deferred"} and str(output.get("text") or "").strip():
            selected.append({**output, "reply_anchor": _reply_anchor(output.get("reply_anchor"))})
    return tuple(selected)


def _utf8_ref(value: Any, label: str) -> str:
    text = str(value or "").strip()
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PresenceHostError(f"presence Host returned non-UTF-8 {label}") from exc
    return text


def _reply_anchor(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("message"), dict):
        return None
    source = str(value.get("source_event_id") or "").strip()
    return {"source_event_id": source, "message": dict(value["message"])} if source else None


def _child_ref(response: Dict[str, Any], polled: str) -> str:
    original = {polled, str(response.get("turn_ref") or "").strip()}
    for key in ("child_work_ref", "work_ref"):
        value = str(response.get(key) or "").strip()
        if value and value not in original:
            return value
    return ""


def _error_payload(data: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _error_detail(data: bytes) -> str:
    payload = _error_payload(data)
    return str(payload.get("error") or payload.get("code") or "request rejected")[:300]
