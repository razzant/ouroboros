"""Shared httpx transport construction for remote LLM clients.

Extracted from ``llm.py`` (size-ratchet byte budget, same precedent as
``loop_transport.py``): one factory owns the TCP-keepalive socket options
for every remote httpx client class, so a NAT/VPN mapping silently dropped
during a long silent reasoning stretch is detected by kernel probes within
minutes instead of hanging until the transport read timeout. Linux and
Darwin both get the idle/interval/count tuning where CPython exports the
constants (``platform_layer.tcp_keepalive_socket_options``); proxy-routed
installs (no explicit transport), the Anthropic-native ``requests`` lane and
every other platform, Windows included (``SO_KEEPALIVE`` only), keep their
current behaviour — a disclosed residual.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def remote_httpx_transport(
    async_client: bool = False,
    *,
    trust_env: bool = True,
    limits: Optional[Any] = None,
):
    """Build the shared keepalive (Async)HTTPTransport.

    Socket options live on the transport (httpx ignores ``socket_options``
    on the Client itself). ``trust_env`` must be forwarded here too: httpx
    uses an explicit transport as-is, so ``Client(trust_env=False)`` alone
    never reaches ``create_ssl_context`` — the no-proxy clients pass
    ``trust_env=False`` to keep SSL_CERT_FILE/SSL_CERT_DIR env isolation.
    """
    import httpx

    from ouroboros.platform_layer import tcp_keepalive_socket_options

    kwargs: Dict[str, Any] = {
        "socket_options": tcp_keepalive_socket_options(),
        "trust_env": trust_env,
    }
    if limits is not None:
        kwargs["limits"] = limits
    transport_cls = httpx.AsyncHTTPTransport if async_client else httpx.HTTPTransport
    try:
        return transport_cls(**kwargs)
    except TypeError:
        # httpx < 0.25 has no ``socket_options`` parameter; the dependency pin
        # is a bare ``httpx``, so such a venv is legal. Rebuild without the
        # keepalive tuning (silently absent) rather than killing every remote
        # call at client construction — same guard shape as the openai
        # Default-client getattr fallback below.
        kwargs.pop("socket_options", None)
        return transport_cls(**kwargs)


def _sdk_pool_limits():
    """openai-SDK-equivalent pool limits for long-lived clients.

    An explicit transport ignores the Client-level limits, silently
    downgrading the SDK's 1000/100 pool to the httpx 100/20 defaults — and a
    self-inflicted PoolTimeout would then read as a transport outage.
    """
    import httpx

    return httpx.Limits(
        max_connections=1000, max_keepalive_connections=100, keepalive_expiry=5.0
    )


def env_proxies_configured() -> bool:
    """True when any proxy httpx would honor is configured.

    Mirrors httpx (``get_environment_proxies`` builds its mounts from
    ``urllib.request.getproxies()``): that includes not only the
    HTTP(S)_PROXY/ALL_PROXY env vars but also macOS SystemConfiguration and
    the Windows registry, so a system-proxy install (no env vars; a typical
    GUI-launched macOS app) is detected too. httpx honors those proxies only
    when no explicit transport is passed, so attaching the keepalive
    transport there would silently break the install's only working egress.
    A lone ``no``/NO_PROXY entry does not count.
    """
    import urllib.request

    proxies = urllib.request.getproxies()
    return any(scheme in ("http", "https", "all") for scheme in proxies)


def keepalive_http_client(async_client: bool = False):
    """openai Default(Async)HttpxClient on the keepalive transport, or None.

    None on proxy-routed installs: they keep the SDK default construction so
    httpx env-proxy mounts survive (disclosed residual: no TCP-keepalive
    tuning there).
    """
    if env_proxies_configured():
        return None
    import openai

    cls = getattr(
        openai,
        "DefaultAsyncHttpxClient" if async_client else "DefaultHttpxClient",
        None,
    )
    if cls is None:
        # An SDK build without the Default client classes falls back to SDK
        # default construction (no keepalive tuning) rather than failing the
        # LLM client construction over a tuning concern.
        return None
    try:
        return cls(
            transport=remote_httpx_transport(async_client, limits=_sdk_pool_limits())
        )
    except TypeError:
        # A future SDK generation whose Default client rejects the transport
        # object falls back to SDK default construction the same way.
        return None


def make_no_proxy_client(target: Dict[str, Any], timeout: Any) -> Tuple[Any, Any]:
    """Per-call OpenAI client fully isolated from proxy/SSL environment.

    ``timeout`` is the caller's explicit per-call bound (an ``httpx.Timeout``
    or a float); no default is applied here.
    """
    import httpx
    from openai import OpenAI

    http_client = httpx.Client(
        trust_env=False,
        mounts={},
        timeout=timeout,
        transport=remote_httpx_transport(trust_env=False),
    )
    oa_client = OpenAI(
        api_key=str(target.get("api_key") or ""),
        base_url=str(target.get("base_url") or ""),
        default_headers=dict(target.get("default_headers") or {}),
        http_client=http_client,
        max_retries=0,
    )
    return oa_client, http_client


def make_no_proxy_async_client(target: Dict[str, Any], timeout: Any) -> Tuple[Any, Any]:
    """Async variant of :func:`make_no_proxy_client`.

    ``timeout`` is the caller's explicit per-call bound, as above.
    """
    import httpx
    from openai import AsyncOpenAI

    http_client = httpx.AsyncClient(
        trust_env=False,
        mounts={},
        timeout=timeout,
        transport=remote_httpx_transport(async_client=True, trust_env=False),
    )
    oa_client = AsyncOpenAI(
        api_key=str(target.get("api_key") or ""),
        base_url=str(target.get("base_url") or ""),
        default_headers=dict(target.get("default_headers") or {}),
        http_client=http_client,
        max_retries=0,
    )
    return oa_client, http_client


def web_search_openai_client(
    *, api_key: str, base_url: Optional[str], timeout: Optional[float] = None,
    default_headers: Optional[Dict[str, str]] = None,
):
    """Web-search OpenAI client (Q16 coverage) on the keepalive transport."""
    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    if default_headers:
        kwargs["default_headers"] = dict(default_headers)
    http_client = keepalive_http_client()
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)
