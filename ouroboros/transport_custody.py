"""Typed transport facts for the physical-attempt custody seam."""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)


def is_loopback_base_url(base_url: Any) -> bool:
    """True when the configured route targets this very host.

    A loopback OPENAI_COMPATIBLE_BASE_URL (the documented Ollama / LM Studio /
    vLLM setups) is a LOCAL server even though its provider name is not
    "local": its connect failure means that server is down, not that the
    network egress is — so such routes must never classify as a remote
    transport outage worth waiting out, nor earn a paid repeat. Loopback is
    the whole class: ``localhost`` by name (one trailing dot tolerated, as the
    resolver does), every 127.0.0.0/8 address in every form ``inet_aton``
    accepts — the dotted quad, the ``127.1`` / ``127.0.1`` shorthands, decimal
    ``2130706433``, hex ``0x7f000001``, octal ``0177.0.0.1`` — ``::1`` and the
    IPv4-mapped IPv6 form; any other name stays remote.
    """
    text = str(base_url or "").strip()
    if not text:
        return False
    try:
        host = (urlsplit(text).hostname or "").lower()  # IPv6 brackets already stripped
    except ValueError:
        return False
    host = host.removesuffix(".")
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)  # exact dotted quad, IPv6, IPv4-mapped
    except ValueError:
        try:
            # The OS accepts every inet_aton spelling for a local server's URL; so must the class.
            address = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return False
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(address.is_loopback or (mapped is not None and mapped.is_loopback))


def is_pre_dispatch_transport_failure(exc: BaseException) -> bool:
    """Return true only for exceptions raised before request bytes can be sent."""
    try:
        import httpx

        # ProxyError is tunnel establishment (CONNECT/SOCKS) failing before any
        # provider request exists — pre-dispatch by construction, like connects.
        safe_types = (
            httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
            httpx.ProxyError,
        )
    except Exception:  # pragma: no cover - httpx ships with the runtime
        return False
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, safe_types):
            return True
        # Only an explicit ``raise ... from ...`` chain carries transport
        # provenance.  Implicit ``__context__`` also links a later fallback
        # exception to the previous provider leg, which would misclassify a
        # dispatched read timeout as a pre-dispatch connect failure.
        current = current.__cause__
    try:
        import requests
        import urllib3

        if isinstance(exc, requests.exceptions.ConnectTimeout):
            return True
        if not isinstance(exc, requests.exceptions.ConnectionError):
            return False
        # requests.exceptions.ProxyError subclasses ConnectionError; both the
        # direct and the proxied connect failure arrive as MaxRetryError args.
        for value in getattr(exc, "args", ()):
            if isinstance(value, urllib3.exceptions.MaxRetryError):
                reason = getattr(value, "reason", None)
                if isinstance(reason, urllib3.exceptions.ConnectTimeoutError):
                    return True
                if isinstance(reason, urllib3.exceptions.ProxyError):
                    # An unreachable proxy is a pre-dispatch fact only with
                    # nested connect-time evidence (NewConnectionError is a
                    # ConnectTimeoutError subclass); a proxy HTTP response or
                    # a post-dispatch read failure never matches.
                    nested = getattr(reason, "original_error", None)
                    if isinstance(nested, (
                        urllib3.exceptions.ConnectTimeoutError,
                        urllib3.exceptions.NewConnectionError,
                    )):
                        return True
    except Exception:  # pragma: no cover - optional transport dependency
        pass
    return False


def _capture_on_chain(error: BaseException) -> Any:
    """The physical-attempt capture riding ``error`` or its explicit causes.

    Deliberately read off the exception chain, NOT the contextvar helper
    (physical_attempt_capture_from_exception's fallback): a custody fact must
    never bind a stale attempt from an unrelated call. Wrappers
    (LocalContextTooLargeError, recovery RuntimeError) carry the capture only
    on their explicit cause — walk it the same way.
    """
    capture = getattr(error, "physical_attempt_capture", None)
    seen: set = set()
    walker = getattr(error, "__cause__", None)
    while capture is None and isinstance(walker, BaseException) and id(walker) not in seen:
        seen.add(id(walker))
        capture = getattr(walker, "physical_attempt_capture", None)
        walker = walker.__cause__
    return capture


def _requests_protocol_death(exc: BaseException) -> Any:
    """The innermost typed death inside a requests body-disconnect wrapper, or None.

    Both of requests' wrappers for a socket that died mid-request count, because
    the Anthropic-native lane's non-streaming ``requests.post`` reads the body
    inside the call: the ``ConnectionError`` raised while the request is in
    flight, and the ``ChunkedEncodingError`` ``Response.iter_content`` raises
    when urllib3 signals a ``ProtocolError`` while the BODY is being read.
    ``ChunkedEncodingError`` does NOT subclass ``ConnectionError`` (its MRO goes
    straight to ``RequestException``), so it has to be named here; it can only
    carry a body-read failure, so the walk below decides it exactly as before.

    requests wraps the urllib3 ``ProtocolError`` (whose own args carry the
    ``RemoteDisconnected``) as the wrapper's first argument, and urllib3 keeps a
    wrapped failure as ``reason`` on ``MaxRetryError`` — none of it is on
    ``__cause__``. The deepest match wins so the durable cause type is the most
    specific fact (``RemoteDisconnected`` over ``ProtocolError``). A
    proxy-tunnel failure (a requests or urllib3 ``ProxyError`` anywhere on the
    walk) is never a death, whatever it wraps: the tunnel, not the provider
    request, is what died, and that class keeps the base no-resend terminal.
    """
    try:
        import http.client
        import requests
        import urllib3
    except Exception:  # pragma: no cover - optional transport dependency
        return None
    if not isinstance(exc, (
        requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
    )) or isinstance(
        exc, (requests.exceptions.Timeout, requests.exceptions.ProxyError),
    ):
        return None
    found = None
    pending = list(getattr(exc, "args", ()))
    while pending:
        value = pending.pop(0)
        if isinstance(value, (urllib3.exceptions.ProxyError, requests.exceptions.ProxyError)):
            return None
        if isinstance(value, (urllib3.exceptions.ProtocolError, http.client.RemoteDisconnected)):
            found = value
        if isinstance(value, BaseException):
            pending.extend((*getattr(value, "args", ()), getattr(value, "reason", None)))
    return found


def is_retryable_transport_death(exc: BaseException) -> bool:
    """True when a DISPATCHED request died with a typed transport death that a
    bounded paid repeat (a NEW physical attempt) may follow.

    The class is deliberately narrow: httpx ``ReadError`` / ``WriteError`` /
    ``RemoteProtocolError`` reached through the explicit ``__cause__`` chain the
    OpenAI SDK sets (``raise APIConnectionError(request=request) from err``), or
    the requests/urllib3 Anthropic-native shape — a ``ConnectionError`` or a
    body-read ``ChunkedEncodingError`` carrying
    ``urllib3.exceptions.ProtocolError`` / ``http.client.RemoteDisconnected``.
    NOT a timeout of any kind (a ``ReadTimeout`` is "we gave up", the provider
    may still be working), NOT a provider status/body error, NOT a pre-dispatch
    failure (that custody is ``released`` and owned by the free wait episode),
    and — the classifier's locality gate — NOT a local provider or a loopback
    route, whose dead server is not a network fault worth paying for again. A
    missing capture proves nothing and fails closed.
    """
    if is_pre_dispatch_transport_failure(exc):
        return False  # the free released class: the two predicates are never both true
    capture = _capture_on_chain(exc)
    if (
        capture is None
        or str(getattr(capture, "state", "") or "") not in ("dispatched", "unresolved")
        or str(getattr(capture, "provider", "") or "") == "local"
        or bool(getattr(capture, "route_is_loopback", False))
    ):
        return False
    try:
        import httpx

        death_types: tuple = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)
    except Exception:  # pragma: no cover - httpx ships with the runtime
        death_types = ()
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, death_types) or _requests_protocol_death(current) is not None:
            return True
        current = current.__cause__
    return False


def release_pre_dispatch_attempt(reservation: Any, exc: BaseException) -> bool:
    """Release a marked attempt only after a typed pre-dispatch transport fact."""
    if not is_pre_dispatch_transport_failure(exc):
        return False
    from ouroboros.usage_accounting import _transition

    try:
        _transition(
            reservation,
            "released",
            _allow_dispatched_release=True,
            reason=f"before_dispatch_failed:{type(exc).__name__}",
        )
    except Exception:
        log.exception("Failed to release pre-dispatch physical attempt")
        return False
    return True


def attempt_custody_event_fields(error: BaseException) -> dict:
    """Additive custody binding for durable error events (nanny-leaf S3).

    The physical-attempt capture already rides the exception; without these
    fields the durable ``llm_api_error`` row cannot be joined to the attempt
    ledger, and the transport class of a wrapped cause (ConnectError vs
    RemoteProtocolError) is unrecoverable after the fact. Bounded type names
    only — never raw cause text.
    """
    capture = _capture_on_chain(error)
    fields: dict = {}
    if capture is not None:
        fields["physical_attempt_id"] = str(getattr(capture, "attempt_id", "") or "")
        fields["attempt_custody_state"] = str(getattr(capture, "state", "") or "")
        provider_error_type = str(getattr(capture, "provider_error_type", "") or "")
        if provider_error_type:
            fields["provider_error_type"] = provider_error_type
    # The same walk the death predicate uses: the explicit ``__cause__`` chain,
    # plus — at every link including the error itself, because the requests lane
    # raises its ConnectionError bare — the typed death requests keeps in
    # ``args``/``reason`` rather than on ``__cause__``.
    seen = set()
    current: Any = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        death = _requests_protocol_death(current)
        module = type(current).__module__ or ""
        if death is not None:
            fields["transport_cause_type"] = type(death).__name__
            break
        if current is not error and (module.split(".")[0] in (
            "httpx", "httpcore", "requests", "urllib3", "ssl", "socket", "anyio",
        ) or (module == "builtins" and isinstance(current, (ConnectionError, TimeoutError)))):
            fields["transport_cause_type"] = type(current).__name__
            break
        current = current.__cause__
    return fields
