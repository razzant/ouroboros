"""Public readiness probes for the Telegram Mini App gateway."""
from __future__ import annotations

import asyncio
import enum
import ipaddress
import json
import socket
from collections.abc import Awaitable
from typing import Any, TypeVar
from urllib.parse import urlsplit

import httpx

from cloudflare_tunnel import QuickTunnel
from sidecar import GATEWAY_MARKER_HEADER


_TRYCLOUDFLARE_SUFFIX = ".trycloudflare.com"
_DOH_ENDPOINT = "https://1.1.1.1/dns-query"
_DOH_HOST = "cloudflare-dns.com"
_MAX_DOH_RESPONSE_BYTES = 32 * 1024
_MAX_DOH_ANSWERS = 64
_MAX_DOH_ADDRESSES = 4
_MAX_GATEWAY_RESPONSE_BYTES = 64 * 1024
_PUBLIC_OPERATION_TIMEOUT_SEC = 4.0
_DOH_RETRY_INTERVAL_SEC = 2.0
_MAX_DOH_ROUNDS = 8
_T = TypeVar("_T")


class _PublicProbeError(RuntimeError):
    pass


class _PublicHostAbsentError(_PublicProbeError):
    """The public resolver returned NXDOMAIN for the queried name itself."""


class PublicGatewayError(RuntimeError):
    pass


CompanionError = PublicGatewayError


def require_running(stop_event: asyncio.Event) -> None:
    if stop_event.is_set():
        raise PublicGatewayError("Companion shutdown was requested.")


class PublicProbeOutcome(enum.Enum):
    READY = "ready"
    UNHEALTHY_RESPONSE = "unhealthy_response"
    HOST_ABSENT = "host_absent"
    OBSERVER_UNAVAILABLE = "observer_unavailable"
    TUNNEL_EXITED = "tunnel_exited"


def _quick_tunnel_host(public_url: str) -> str:
    try:
        parsed = urlsplit(str(public_url or "").strip())
        port = parsed.port
    except ValueError as exc:
        raise CompanionError("Public tunnel URL is invalid.") from exc
    host = str(parsed.hostname or "").lower()
    label = host[: -len(_TRYCLOUDFLARE_SUFFIX)] if host.endswith(_TRYCLOUDFLARE_SUFFIX) else ""
    if (
        parsed.scheme.lower() != "https"
        or parsed.netloc != host
        or port is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not label
        or "." in label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label)
    ):
        raise CompanionError("Public tunnel URL is not an exact Cloudflare Quick Tunnel origin.")
    return host


def _is_dns_resolution_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return True
        for value in current.args:
            if isinstance(value, socket.gaierror):
                return True
        current = current.__cause__ or current.__context__
    return False


async def _run_startup_operation(
    operation: Awaitable[_T],
    *,
    stop_event: asyncio.Event | None,
    timeout_sec: float,
) -> _T:
    """Bound one startup operation and make graceful shutdown win every race."""
    operation_task = asyncio.create_task(operation)
    stop_task = asyncio.create_task(stop_event.wait()) if stop_event is not None else None
    waiters: set[asyncio.Task[Any]] = {operation_task}
    if stop_task is not None:
        waiters.add(stop_task)
    try:
        done, _pending = await asyncio.wait(
            waiters,
            timeout=max(0.01, float(timeout_sec)),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task is not None and stop_task in done:
            operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            require_running(stop_event)
        if operation_task in done:
            result = await operation_task
            if stop_event is not None:
                require_running(stop_event)
            return result
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise asyncio.TimeoutError
    finally:
        if stop_task is not None:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)


async def _gateway_response_ready(response: httpx.Response) -> bool:
    if response.status_code != 200 or response.headers.get(GATEWAY_MARKER_HEADER) != "1":
        return False
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > _MAX_GATEWAY_RESPONSE_BYTES:
            return False
    return True


async def _probe_gateway_normally(
    client: httpx.AsyncClient,
    public_url: str,
    *,
    timeout_sec: float,
) -> bool:
    async with client.stream(
        "GET",
        public_url,
        headers={
            "Accept": "text/html",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
        },
        timeout=timeout_sec,
    ) as response:
        return await _gateway_response_ready(response)


def _is_public_unicast_ipv4(address: ipaddress.IPv4Address) -> bool:
    return address.is_global and not (
        address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_private
    )


def _parse_doh_ipv4_response(raw: bytes, host: str) -> tuple[str, ...]:
    if len(raw) > _MAX_DOH_RESPONSE_BYTES:
        raise _PublicProbeError("Cloudflare DNS response was oversized.")
    try:
        payload = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise _PublicProbeError("Cloudflare DNS response was invalid.") from exc
    if not isinstance(payload, dict):
        raise _PublicProbeError("Cloudflare DNS response had an unexpected shape.")
    status = payload.get("Status")
    if type(status) is not int or status not in (0, 3) or payload.get("TC") is not False:
        raise _PublicProbeError("Cloudflare DNS did not return a complete address or name-error response.")
    questions = payload.get("Question")
    if not isinstance(questions, list) or len(questions) != 1 or not isinstance(questions[0], dict):
        raise _PublicProbeError("Cloudflare DNS response did not bind one question.")
    question = questions[0]
    name = question.get("name")
    qtype = question.get("type")
    qclass = question.get("class", 1)  # Cloudflare's DNS JSON omits the implicit Internet class.
    if (
        not isinstance(name, str) or not name.isascii() or name.lower().removesuffix(".") != host
        or type(qtype) is not int or qtype != 1
        or type(qclass) is not int or qclass != 1
    ):
        raise _PublicProbeError("Cloudflare DNS response did not match the tunnel's IN A question.")
    answers = payload.get("Answer", [])
    if not isinstance(answers, list) or len(answers) > _MAX_DOH_ANSWERS:
        raise _PublicProbeError("Cloudflare DNS response had an invalid answer set.")
    if status == 3:
        # A CNAME can exist while its target is absent. Authority records (e.g. SOA)
        # accompany ordinary negative answers and do not assert that the host exists.
        if answers:
            raise _PublicProbeError("NXDOMAIN with answer records did not establish absence of the queried name.")
        raise _PublicHostAbsentError("Cloudflare DNS returned NXDOMAIN for the tunnel hostname.")
    addresses: list[str] = []
    for answer in answers:
        if (
            not isinstance(answer, dict)
            or type(answer.get("type")) is not int
            or answer.get("type") != 1
        ):
            continue
        try:
            address = ipaddress.IPv4Address(str(answer.get("data") or ""))
        except ipaddress.AddressValueError:
            continue
        if not _is_public_unicast_ipv4(address):
            continue
        value = str(address)
        if value not in addresses:
            addresses.append(value)
            if len(addresses) >= _MAX_DOH_ADDRESSES:
                break
    if not addresses:
        raise _PublicProbeError("Cloudflare DNS returned no usable public IPv4 address.")
    return tuple(addresses)


async def _resolve_public_ipv4_via_doh(host: str, *, timeout_sec: float) -> tuple[str, ...]:
    async with httpx.AsyncClient(
        timeout=timeout_sec,
        follow_redirects=False,
        trust_env=False,
        http2=False,
    ) as client:
        async with client.stream(
            "GET",
            _DOH_ENDPOINT,
            params={"name": host, "type": "A"},
            headers={
                "Accept": "application/dns-json",
                "Accept-Encoding": "identity",
                "Host": _DOH_HOST,
            },
            extensions={"sni_hostname": _DOH_HOST},
            timeout=timeout_sec,
        ) as response:
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if response.status_code != 200 or content_type != "application/dns-json":
                raise _PublicProbeError("Cloudflare DNS-over-HTTPS request failed.")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_DOH_RESPONSE_BYTES:
                    raise _PublicProbeError("Cloudflare DNS response was oversized.")
    return _parse_doh_ipv4_response(bytes(body), host)


async def _probe_gateway_via_ipv4(
    host: str,
    address: str,
    *,
    timeout_sec: float,
) -> bool:
    try:
        parsed_address = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError as exc:
        raise _PublicProbeError("Cloudflare DNS returned an invalid IPv4 address.") from exc
    if not _is_public_unicast_ipv4(parsed_address):
        raise _PublicProbeError("Cloudflare DNS returned a non-public IPv4 address.")
    async with httpx.AsyncClient(
        timeout=timeout_sec,
        follow_redirects=False,
        trust_env=False,
        http2=False,
    ) as client:
        async with client.stream(
            "GET",
            f"https://{parsed_address}/",
            headers={
                "Accept": "text/html",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache",
                "Host": host,
            },
            extensions={"sni_hostname": host},
            timeout=timeout_sec,
        ) as response:
            return await _gateway_response_ready(response)


async def probe_public_gateway_once(
    public_url: str,
    tunnel: QuickTunnel,
    *,
    timeout_sec: float = _PUBLIC_OPERATION_TIMEOUT_SEC,
    stop_event: asyncio.Event | None = None,
) -> PublicProbeOutcome:
    if stop_event is not None:
        require_running(stop_event)
    if tunnel.returncode is not None:
        return PublicProbeOutcome.TUNNEL_EXITED
    host = _quick_tunnel_host(public_url)
    async with httpx.AsyncClient(
        timeout=timeout_sec,
        follow_redirects=False,
        trust_env=False,
        http2=False,
    ) as client:
        try:
            ready = await _run_startup_operation(
                _probe_gateway_normally(client, public_url, timeout_sec=timeout_sec),
                stop_event=stop_event,
                timeout_sec=timeout_sec,
            )
        except httpx.HTTPError as exc:
            if not _is_dns_resolution_error(exc):
                return PublicProbeOutcome.OBSERVER_UNAVAILABLE
        except (asyncio.TimeoutError, OSError):
            return PublicProbeOutcome.OBSERVER_UNAVAILABLE
        else:
            return (
                PublicProbeOutcome.READY
                if ready and tunnel.returncode is None
                else PublicProbeOutcome.UNHEALTHY_RESPONSE
            )

    try:
        addresses = await _run_startup_operation(
            _resolve_public_ipv4_via_doh(host, timeout_sec=timeout_sec),
            stop_event=stop_event,
            timeout_sec=timeout_sec,
        )
    except _PublicHostAbsentError:
        return PublicProbeOutcome.HOST_ABSENT
    except (httpx.HTTPError, asyncio.TimeoutError, OSError, _PublicProbeError):
        return PublicProbeOutcome.OBSERVER_UNAVAILABLE
    saw_response = False
    for address in addresses:
        if tunnel.returncode is not None:
            return PublicProbeOutcome.TUNNEL_EXITED
        try:
            ready = await _run_startup_operation(
                _probe_gateway_via_ipv4(host, address, timeout_sec=timeout_sec),
                stop_event=stop_event,
                timeout_sec=timeout_sec,
            )
        except (httpx.HTTPError, asyncio.TimeoutError, OSError, _PublicProbeError):
            continue
        saw_response = True
        if ready and tunnel.returncode is None:
            return PublicProbeOutcome.READY
    return (
        PublicProbeOutcome.UNHEALTHY_RESPONSE
        if saw_response
        else PublicProbeOutcome.OBSERVER_UNAVAILABLE
    )


async def verify_public_gateway(
    public_url: str,
    tunnel: QuickTunnel,
    *,
    timeout_sec: float = 35.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.01, float(timeout_sec))
    doh_rounds = 0
    while True:
        if stop_event is not None:
            require_running(stop_event)
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise CompanionError("Public tunnel did not reach the authentication sidecar.")
        outcome = await probe_public_gateway_once(
            public_url,
            tunnel,
            timeout_sec=min(_PUBLIC_OPERATION_TIMEOUT_SEC, remaining),
            stop_event=stop_event,
        )
        if outcome is PublicProbeOutcome.READY:
            return
        if outcome is PublicProbeOutcome.TUNNEL_EXITED:
            raise CompanionError(
                f"Cloudflared exited before public verification (exit {tunnel.returncode})."
            )
        if outcome is PublicProbeOutcome.OBSERVER_UNAVAILABLE:
            doh_rounds += 1
            if doh_rounds >= _MAX_DOH_ROUNDS and loop.time() >= deadline:
                raise CompanionError("Public tunnel could not be observed safely.")
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise CompanionError("Public tunnel did not reach the authentication sidecar.")
        if stop_event is None:
            await asyncio.sleep(min(_DOH_RETRY_INTERVAL_SEC, remaining))
        else:
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=min(_DOH_RETRY_INTERVAL_SEC, remaining),
                )
            except asyncio.TimeoutError:
                pass
