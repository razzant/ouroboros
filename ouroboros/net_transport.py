"""Shared httpx transport construction for remote LLM clients.

Extracted from ``llm.py`` (size-ratchet byte budget, same precedent as
``loop_transport.py``): one factory owns the TCP-keepalive socket options
for every remote httpx client class, so a NAT/VPN mapping silently dropped
during a long silent reasoning stretch is detected by kernel probes within
minutes instead of hanging until the transport read timeout. Linux and
Darwin both get the idle/interval/count tuning where CPython exports the
constants, with values from ``config.py`` and Darwin's idle threshold set via
``TCP_KEEPALIVE`` (``platform_layer.tcp_keepalive_socket_options``); proxy-routed
installs (the cached and web-search clients skip the explicit transport, because
httpx env-proxy mounts require it absent), the Anthropic-native ``requests`` lane and
every other platform, Windows included (``SO_KEEPALIVE`` only), keep their
current behaviour — a disclosed residual.

The same module owns the trust bundle: ``extra_ca_bundle`` merges the owner's
``OUROBOROS_EXTRA_CA_BUNDLE`` PEM over certifi once and every first-party
client (httpx transports here, the Anthropic ``requests`` lane, the GigaChat
SDK, catalog and probe clients) verifies against that one path.

It also owns how an openai SDK client authenticates: ``openai_client_class``
gives a route the routing leaf declared keyless the SDK's keyless form.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple


class ExtraCaBundleError(RuntimeError):
    """``OUROBOROS_EXTRA_CA_BUNDLE`` names a file that cannot serve as a trust anchor."""


_EXTRA_CA_BUNDLE_KEY = "OUROBOROS_EXTRA_CA_BUNDLE"
_merged_bundle_cache: Dict[Tuple[str, str, int, int], str] = {}
_ssl_context_cache: Dict[str, Any] = {}


def extra_ca_bundle() -> Optional[str]:
    """Path of the trust bundle every first-party HTTP client verifies against, or None.

    ``OUROBOROS_EXTRA_CA_BUNDLE`` names a PEM file with the CA certificates the
    default bundle lacks (a TLS-inspecting corporate proxy, a national CA such
    as the one behind GigaChat). httpx, requests and the GigaChat SDK each take
    ONE bundle path and treat it as the whole trust list, so the owner's file is
    merged with certifi into a content-addressed
    ``<data>/state/extra-ca-bundle/<digest>.pem`` and that path is returned: a
    changed owner file yields a new path, so every cache keyed on the path — the
    SSL context below and the provider clients — rotates with it; siblings older
    than a day are pruned (a task still holding an earlier setting keeps its file).
    Unset returns None and every client is built exactly as before the setting
    existed. An unreadable or non-PEM file raises ``ExtraCaBundleError``: a
    silent fall-back to certifi would reproduce the very TLS failure the owner
    set the key to cure.
    """
    from ouroboros.settings_integrity import runtime_setting

    raw = str(runtime_setting(_EXTRA_CA_BUNDLE_KEY, "") or "").strip()
    if not raw:
        return None
    import os
    import pathlib

    from ouroboros.config import DATA_DIR

    extra = pathlib.Path(raw).expanduser()
    bundle_dir = pathlib.Path(DATA_DIR) / "state" / "extra-ca-bundle"
    try:
        stat = extra.stat()
    except OSError as exc:
        raise ExtraCaBundleError(f"{_EXTRA_CA_BUNDLE_KEY} is not readable: {extra} ({exc})") from exc
    key = (str(extra), str(bundle_dir), stat.st_mtime_ns, stat.st_size)
    cached = _merged_bundle_cache.get(key)
    if cached and os.path.isfile(cached):
        return cached
    import ssl

    import certifi

    try:
        extra_bytes = extra.read_bytes()
    except OSError as exc:
        raise ExtraCaBundleError(f"{_EXTRA_CA_BUNDLE_KEY} is not readable: {extra} ({exc})") from exc
    try:
        # Load the owner's file on its own first: a malformed PEM must surface as
        # this typed error at the setting, not as an SSLError inside some client.
        ssl.create_default_context().load_verify_locations(cadata=extra_bytes.decode("ascii", "replace"))
    except (ssl.SSLError, ValueError) as exc:
        raise ExtraCaBundleError(f"{_EXTRA_CA_BUNDLE_KEY} holds no loadable PEM certificate: {extra} ({exc})") from exc

    import hashlib
    import time

    from ouroboros.utils import write_bytes_atomic

    base = pathlib.Path(certifi.where()).read_bytes()
    merged = base.rstrip(b"\n") + b"\n" + extra_bytes.rstrip(b"\n") + b"\n"
    digest = hashlib.sha256(merged).hexdigest()[:12]
    target = bundle_dir / f"{digest}.pem"
    try:
        if not target.is_file():
            bundle_dir.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(target, merged)
        cutoff = time.time() - 86400
        for stale in bundle_dir.glob("*.pem"):
            if stale != target:
                try:
                    if stale.stat().st_mtime < cutoff:
                        stale.unlink()
                except OSError:
                    pass  # a sibling another process is materializing or has just pruned
    except OSError as exc:
        raise ExtraCaBundleError(f"cannot write the merged trust bundle {target}: {exc}") from exc
    _merged_bundle_cache[key] = str(target)
    return str(target)


def trust_ssl_context():
    """The ``ssl.SSLContext`` over the merged bundle, or None without the setting.

    httpx deprecates a path-valued ``verify``; one context per merged bundle is
    built here and shared by every httpx client (a context is read-only after
    construction, so sharing across clients and threads is safe).
    """
    bundle = extra_ca_bundle()
    if bundle is None:
        return None
    context = _ssl_context_cache.get(bundle)
    if context is None:
        import ssl

        context = ssl.create_default_context(cafile=bundle)
        _ssl_context_cache[bundle] = context
    return context


def verify_kwargs() -> Dict[str, Any]:
    """``{"verify": <SSLContext>}`` for an httpx constructor, ``{}`` without the setting."""
    context = trust_ssl_context()
    return {"verify": context} if context is not None else {}


def requests_verify_kwargs() -> Dict[str, Any]:
    """``{"verify": <bundle path>}`` for a ``requests`` call, ``{}`` without the setting."""
    bundle = extra_ca_bundle()
    return {"verify": bundle} if bundle else {}


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
    kwargs.update(verify_kwargs())
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


def keepalive_http_client(async_client: bool = False, *, follow_redirects: bool = True):
    """openai Default(Async)HttpxClient on the keepalive transport, or None.

    None on proxy-routed installs without a trust bundle: they keep the SDK
    default construction so httpx env-proxy mounts survive (disclosed residual:
    no TCP-keepalive tuning there); with a bundle the Default client carries it
    as ``verify`` and still mounts the env proxies. ``follow_redirects=False`` is
    held by the client itself (openai<2 has no per-request redirect option), so a
    proxy-routed install then gets the Default client without a transport, not None.
    """
    proxied = env_proxies_configured()
    bundle = extra_ca_bundle()
    if proxied and bundle is None and follow_redirects:
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
    # Proxy-routed installs keep SDK default construction (env proxy mounts
    # survive only without an explicit transport) and receive the trust bundle
    # through ``verify``; everyone else gets it on the keepalive transport.
    kwargs: Dict[str, Any] = (
        verify_kwargs() if proxied
        else {"transport": remote_httpx_transport(async_client, limits=_sdk_pool_limits())}
    )
    if not follow_redirects:
        kwargs["follow_redirects"] = False
    try:
        return cls(**kwargs)
    except TypeError:
        # A future SDK generation whose Default client rejects the transport
        # object falls back to SDK default construction the same way.
        return None


_keyless_client_classes: Dict[Any, Any] = {}


def openai_client_class(target: Dict[str, Any], *, async_client: bool = False):
    """The openai SDK client class for ``target``: the SDK's own, or its keyless form.

    Only the route declares an endpoint keyless (``target["keyless"]``: the dedicated
    OpenAI-compatible endpoint saved without a key, "Leave empty for no auth"); a key on
    the target always wins, and every other target gets the SDK class unchanged. openai 2.x
    refuses a blank key twice before any HTTP: construction raises "Missing credentials"
    and each request "Could not resolve authentication method". The keyless form waives
    the construction check and sends the SDK's own explicit ``Authorization`` omission
    with every request, so no credential from any source (a key, the SDK's environment,
    default headers) reaches the endpoint, and no placeholder key exists anywhere.
    """
    import openai

    base = openai.AsyncOpenAI if async_client else openai.OpenAI
    if not target.get("keyless") or str(target.get("api_key") or ""):
        return base
    keyless = _keyless_client_classes.get(base)
    if keyless is None:
        keyless = _keyless_client_classes[base] = _keyless_client_class(base, async_client)
    return keyless


def _keyless_client_class(base, async_client: bool):
    import inspect

    from openai import Omit

    # openai<2 has no construction check to waive: it already accepts a blank key.
    waiver = ({"_enforce_credentials": False}
              if "_enforce_credentials" in inspect.signature(base.__init__).parameters else {})

    def omit_authorization(options):
        headers = options.headers if isinstance(options.headers, Mapping) else {}
        options.headers = {**headers, "Authorization": Omit()}
        return options

    class Keyless(base):
        def __init__(self, **kwargs):  # ``copy()``/``with_options()`` rebuild through here too
            super().__init__(**{**kwargs, "api_key": "", **waiver})

        def _build_headers(self, options, *, retries_taken=0):
            headers = super()._build_headers(options, retries_taken=retries_taken)
            # SDK dictionaries merge case-sensitively; the final Headers object does not.
            # Strip every spelling, including defaults or per-request overrides.
            headers.pop("Authorization", None)
            return headers

    if async_client:
        class KeylessAsync(Keyless):
            async def _prepare_options(self, options):
                return omit_authorization(await super()._prepare_options(options))
        client = KeylessAsync
    else:
        class KeylessSync(Keyless):
            def _prepare_options(self, options):
                return omit_authorization(super()._prepare_options(options))
        client = KeylessSync

    client.__name__ = client.__qualname__ = f"Keyless{base.__name__}"
    return client


def make_no_proxy_client(target: Dict[str, Any], timeout: Any) -> Tuple[Any, Any]:
    """Per-call OpenAI client fully isolated from proxy/SSL environment.

    ``timeout`` is the caller's explicit per-call bound (an ``httpx.Timeout``
    or a float); no default is applied here.
    """
    import httpx

    http_client = httpx.Client(
        trust_env=False,
        mounts={},
        timeout=timeout,
        transport=remote_httpx_transport(trust_env=False),
    )
    oa_client = openai_client_class(target)(
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

    http_client = httpx.AsyncClient(
        trust_env=False,
        mounts={},
        timeout=timeout,
        transport=remote_httpx_transport(async_client=True, trust_env=False),
    )
    oa_client = openai_client_class(target, async_client=True)(
        api_key=str(target.get("api_key") or ""),
        base_url=str(target.get("base_url") or ""),
        default_headers=dict(target.get("default_headers") or {}),
        http_client=http_client,
        max_retries=0,
    )
    return oa_client, http_client


def web_search_openai_client(
    *, api_key: str, base_url: Optional[str], timeout: Optional[float] = None,
    default_headers: Optional[Dict[str, str]] = None, follow_redirects: bool = True,
):
    """Web-search OpenAI client (Q16 coverage) on the keepalive transport.

    ``follow_redirects=False`` (DeepSeek Search) fails closed on an SDK without a usable Default client."""
    from openai import OpenAI

    kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    if default_headers:
        kwargs["default_headers"] = dict(default_headers)
    http_client = keepalive_http_client(follow_redirects=follow_redirects)
    if http_client is None and not follow_redirects:
        raise RuntimeError("this openai SDK cannot build a web search client that refuses redirects")
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)
