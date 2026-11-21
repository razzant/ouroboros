"""The window an OpenAI-compatible server states about itself, read from its own documented GETs.

Each reader names the server, the documented path and the field (``READERS``, ``MODELS_FIELDS``). The window read
here is the one serving buffer the input and the reply share, per request (llama.cpp's slot ``n_ctx`` is already
divided by ``--parallel``), so it is recorded with source ``server_metadata`` (``context_window._REPLY_INSIDE``).
A gateway catalog's ``context_length`` keeps its older meaning, ``provider_metadata``.

Every failure (refused connection, timeout, non-JSON, another model loaded, a missing or zero field) is ``0``,
"unknown": never a guessed window and never an error the owner sees. Training lengths (llama.cpp ``n_ctx_train``,
Ollama ``/api/show``, LM Studio ``max_context_length``) are what the model could take, not what is loaded, and
are not loaded windows. For client-configured Ollama, /api/show instead bounds the next requested load.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

SOURCE_LOCAL_HEALTH = "local_health"
SOURCE_SERVER_METADATA = "server_metadata"
SOURCE_PROVIDER_METADATA = "provider_metadata"
PHYSICAL_SOURCES = frozenset({SOURCE_SERVER_METADATA, SOURCE_LOCAL_HEALTH})

def physical_statement(row: Dict[str, Any]) -> bool:
    return row.get("source") == SOURCE_LOCAL_HEALTH or (row.get("source") == SOURCE_SERVER_METADATA
                                                       and row.get("provider") == "openai-compatible")



class MetadataWindow(int):
    """A window read from metadata that also names its source (``server_metadata`` or ``provider_metadata``);
    every older caller keeps reading it as the plain number it is."""

    source: str = SOURCE_PROVIDER_METADATA

    def __new__(cls, tokens: int, source: str = SOURCE_PROVIDER_METADATA, instance_id: str = ""):
        window = super().__new__(cls, tokens)
        window.source, window.instance_id = source, instance_id
        return window


READ_TIMEOUT_SEC = 2.0

#: Fields of a ``GET {base_url}/models`` item: (path inside the item, source). The first stated one wins.
MODELS_FIELDS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("max_model_len",), SOURCE_SERVER_METADATA),          # vLLM, SGLang: the served max_model_len
    (("meta", "n_ctx"), SOURCE_SERVER_METADATA),           # llama.cpp: the slot window (server-context.cpp:5654)
    (("context_length",), SOURCE_PROVIDER_METADATA),       # gateway catalogs (the host's reading before this file)
    (("context_window",), SOURCE_PROVIDER_METADATA),
    (("max_context_length",), SOURCE_PROVIDER_METADATA),
    (("meta", "max_model_len"), SOURCE_PROVIDER_METADATA),
    (("meta", "context_length"), SOURCE_PROVIDER_METADATA),
    (("meta", "context_window"), SOURCE_PROVIDER_METADATA),
    (("meta", "max_context_length"), SOURCE_PROVIDER_METADATA),
)


def _positive(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0


def _dig(item: Any, path: Iterable[str]) -> Any:
    for key in path:
        item = item.get(key) if isinstance(item, dict) else None
    return item


def models_item_window(item: Dict[str, Any]) -> Tuple[int, str]:
    """``(window, source)`` one ``/models`` item states, else ``(0, "")``."""
    for path, source in MODELS_FIELDS:
        if window := _positive(_dig(item, path)):
            return window, source
    return 0, ""


def wanted_ids(model: str) -> set:
    """The saved route spelling (``openai-compatible::name``) and the bare id a server lists."""
    from ouroboros.provider_models import parse_model_route

    return {str(model), parse_model_route(model).model}


def _same_model(name: Any, wanted: set) -> bool:
    name = str(name or "")
    return bool(name) and (name in wanted or (name.endswith(":latest") and name[:-len(":latest")] in wanted)
                           or any(f"{w}:latest" == name for w in wanted))


def _llama_props(payload: Any, wanted: set) -> int:
    """llama.cpp ``GET /props``: ``default_generation_settings.n_ctx`` is the slot window; a server serving another
    alias (router mode, a reused port) states nothing about this model."""
    if not isinstance(payload, dict):
        return 0
    alias = payload.get("model_alias")
    if alias and not _same_model(alias, wanted):
        return 0
    return _positive(_dig(payload, ("default_generation_settings", "n_ctx")))


def _lmstudio_v0(payload: Any, wanted: set) -> int:
    """LM Studio ``GET /api/v0/models``: ``loaded_context_length`` of the loaded instance of this model."""
    items = payload.get("data") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and _same_model(item.get("id"), wanted) and item.get("state") == "loaded":
            return _positive(item.get("loaded_context_length"))
    return 0


def ollama_load(payload: Any, wanted: set) -> dict:
    """Ollama ``GET /api/ps``: ``context_length`` the scheduler allocated to the loaded model (0 before a load)."""
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and (_same_model(item.get("name"), wanted) or _same_model(item.get("model"), wanted)):
            return item
    return {}


def _ollama_ps(payload: Any, wanted: set) -> int:
    return _positive(ollama_load(payload, wanted).get("context_length"))


#: Readers at the server root (``base_url`` without its ``/v1``), asked in order when ``/models`` stated no window.
READERS: Tuple[Tuple[str, str, Callable[[Any, set], int]], ...] = (
    ("llama.cpp", "/props", _llama_props),
    ("LM Studio", "/api/v0/models", _lmstudio_v0),
    ("Ollama", "/api/ps", _ollama_ps),
)


def _llama_cpp_document(payload: Any) -> bool:
    """llama.cpp ``GET /props``: its slot settings beside its slot count."""
    return (isinstance(payload, dict) and _positive(_dig(payload, ("default_generation_settings", "n_ctx"))) > 0
            and _positive(payload.get("total_slots")) > 0)


def _lmstudio_document(payload: Any) -> bool:
    """LM Studio ``GET /api/v0/models``: every listed model with its load state and maximum context."""
    items = payload.get("data") if isinstance(payload, dict) else None
    return isinstance(items, list) and bool(items) and all(
        isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        and item.get("state") in ("loaded", "not-loaded") and _positive(item.get("max_context_length")) > 0
        for item in items)


#: A program named by its own documented metadata document at the server root: (program, path, its shape). An
#: error body, a missing path, a version number or a generic ``/v1/models`` listing names no program. The
#: documents these shapes were read from, with their sources: ``tests/fixtures/server_windows/README.md``.
PROGRAM_DOCUMENTS: Tuple[Tuple[str, str, Callable[[Any], bool]], ...] = (
    ("llama.cpp", "/props", _llama_cpp_document),
    ("LM Studio", "/api/v0/models", _lmstudio_document),
)


def server_root(base_url: str) -> str:
    root = str(base_url or "").strip().rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


#: Names no public DNS serves (RFC 6762, RFC 8375, ICANN .internal) and the usual router/compose suffixes.
PRIVATE_SUFFIXES = (".local", ".localhost", ".lan", ".internal", ".home.arpa", ".localdomain")


def host_address(host: str):
    """A numeric host as the address a connection to it reaches, ``None`` for a name: parsed, never resolved.

    Every spelling the OS takes for one address is that address: the dotted quad, the ``10.1`` and ``192.168.1``
    shorthands, a decimal, hex or octal number, and an IPv4-mapped IPv6 address as its IPv4. One reading for whether
    a host is private (``private_host``) and for the address a hold names (``llm_ollama_guard.address_scope``):
    a spelling that one of them took for a name would send past the other."""
    import ipaddress
    import socket

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            address = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            return None
    return getattr(address, "ipv4_mapped", None) or address


def private_host(base_url: str) -> bool:
    """Loopback, private-network, single-label (``ollama``, a compose/LAN name) or private-suffix hosts: the servers the
    root readers are meant for. A public gateway is never asked for another product's paths. A numeric host is judged
    by its address in whatever spelling it was written (``host_address``), before any rule about names."""
    from urllib.parse import urlsplit

    try:
        host = (urlsplit(str(base_url or "").strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    address = host_address(host)
    if address is not None:
        return address.is_loopback or address.is_private or address.is_link_local
    return host == "localhost" or host.endswith(PRIVATE_SUFFIXES) or bool(host and "." not in host and ":" not in host)


def read_root_window(base_url: str, model: str, headers: Optional[Dict[str, str]] = None, *,
                     get: Optional[Callable[..., Any]] = None) -> Tuple[int, str]:
    """``(window, "<server> <path>")`` from the first reader that states one for ``model``, else ``(0, "")``."""
    import httpx

    if get is None:
        from ouroboros.net_transport import verify_kwargs

        def get(url, **kwargs):
            return httpx.get(url, **kwargs, **verify_kwargs())

    root, wanted = server_root(base_url), wanted_ids(model)
    if not root:
        return 0, ""
    for server, path, reader in READERS:
        try:
            response = get(root + path, headers=dict(headers or {}), timeout=READ_TIMEOUT_SEC)
            if getattr(response, "status_code", 0) != 200:
                continue
            window = reader(response.json(), wanted)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return 0, ""  # the same server is unreachable at every root path
        except Exception:
            continue
        if window:
            return window, f"{server} {path}"
    return 0, ""


def local_health_window(model: str) -> int:
    """One live owned instance's window and process binding, never training metadata."""
    try:
        from ouroboros.local_model import get_manager
        evidence = get_manager().serving_context_evidence() or {}
        if evidence.get("confirmed") is True:
            return MetadataWindow(max(0, int(evidence.get("context_window") or 0)), SOURCE_LOCAL_HEALTH,
                                  str(evidence.get("process_id") or ""))
    except Exception:
        pass
    return 0



def read_metadata_window(model: str, base_url: str, api_key: str, *, provider: str, evidence_root,
                         headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """CW6 (v6.34.0): an OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI, ...)
    commonly publishes the per-model window in GET {base_url}/models. ``server_window`` names the fields: vLLM's
    ``max_model_len`` and llama.cpp's ``meta.n_ctx`` are the server's own buffer (``server_metadata``), a catalog's
    ``context_length`` / ``context_window`` stay ``provider_metadata``. When the listing answered without one, a
    server on this host or network is asked at its own documented paths (llama.cpp ``/props``, LM Studio
    ``/api/v0/models``, Ollama ``/api/ps``). Best-effort, fail-closed to ``(0, "")`` (network/auth/parse error,
    no base_url, or hot-path allow_fetch=False).

    The caller supplies the route key and evidence root; this reader never resolves another connection.
    """
    try:
        import httpx

        headers = dict(headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        from ouroboros.net_transport import verify_kwargs
        resp = httpx.get(str(base_url).rstrip("/") + "/models", headers=headers, timeout=5.0, **verify_kwargs())
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if provider == "openai-compatible":  # MiniMax /models carries no modality field
            from ouroboros.vision_routing import record_catalog_image_input
            # Scoped to the key this request sent ("" = none), never one resolved after the response.
            record_catalog_image_input(provider, base_url, items, source="the OpenAI-compatible gateway's /models",
                                       api_key=api_key)
        # The saved model is normally route-prefixed (e.g. ``openai-compatible::llama-3``)
        # while /models lists the BARE id — match either spelling.
        wanted = wanted_ids(model)
        matched = [item for item in (items or []) if isinstance(item, dict)
                   and str(item.get("id") or item.get("name") or "") in wanted]
        from ouroboros.response_limits import positive_limit, record_metadata_limit
        # A listing that was read is an observation even when it publishes no maximum.
        record_metadata_limit(evidence_root, provider=provider, model=model,
            base_url=base_url, maximum=max((positive_limit(item.get("max_output_tokens")) for item in matched), default=0),
            source="/models max_output_tokens")
        for item in matched:
            window, source = models_item_window(item)
            if window:
                return window, source
        if provider == "openai-compatible" and private_host(base_url):
            window, _where = read_root_window(base_url, model, headers)
            if window:
                return window, SOURCE_SERVER_METADATA
        return 0, ""
    except Exception:
        return 0, ""


def ollama_server_ready(target: dict, client: Any) -> Optional[bool]:
    """Fresh recovery evidence; None means a known native address no longer answers as Ollama: another
    program replaced it, or nothing names it and the next send states the hold."""
    from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
    from ouroboros.llm_ollama import OllamaProgramChanged, _current_version, _discoverable, _drive, _url

    if not (_discoverable(target) and target.get("resolved_model") and (target.get("api_key") or target.get("keyless"))):
        return False
    headers = dict(target.get("default_headers") or {})
    if target.get("api_key"):
        headers["Authorization"] = "Bearer " + target["api_key"]
    from ouroboros.llm_ollama_wire import OllamaRouteUnsettledError
    def read(method, path, body):
        if method == "POST":
            return client.post(_url(target, path), headers=headers, json=body)
        return client.get(_url(target, path), headers=headers)

    try:
        if not _drive(_current_version(target), read):
            return False
    except (OllamaProgramChanged, UnidentifiedRouteWithheld):
        return None
    except OllamaRouteUnsettledError:
        return False
    response = read("POST", "/api/show", {"model": target["resolved_model"]})
    shown = response.json() if response.status_code == 200 else None
    if not isinstance(shown, dict) or shown.get("remote_model") or shown.get("remote_host"):
        return False
    architecture = _dig(shown, ("model_info", "general.architecture"))
    capabilities = shown.get("capabilities")
    return bool(isinstance(architecture, str) and architecture.strip()
                and isinstance(capabilities, list) and "completion" in capabilities)


def ollama_window_request(provider, model, base_url, *, api_key, headers, target, allow_fetch=True, force=False, probes=None):
    """Bound native target and selected load. A cache-only reader never discovers a server."""
    from ouroboros.llm_ollama import _discoverable, _known, _identity, is_ollama
    from ouroboros.llm_routing import configured_window
    from ouroboros.utils import transport_no_proxy

    if not _discoverable(target or dict(provider=provider, base_url=base_url)):
        return None
    if target is None:
        target = _ollama_window_target(provider, model, base_url, api_key=api_key, headers=headers)
    if not (target.get("api_key") or target.get("keyless")):
        return None
    identity = _identity(target)
    if target.get("bound_load_window") is not None:
        # Intent captured at task start, usable only if this route later proves native.
        target["requested_load_window"] = target["bound_load_window"]
    if allow_fetch:
        from ouroboros.llm_ollama_guard import UnidentifiedRouteWithheld
        try:
            known = is_ollama(target, no_proxy=transport_no_proxy(), before_send=force)
        except UnidentifiedRouteWithheld:
            known = False  # a window read asks no generation: what withholds a send leaves it to the server's listing
    else:
        known = _known(target, identity) or ("route_notice" not in target and identity in (probes or {}))
    if known:
        target["requested_load_window"] = configured_window(target)
        return target, target["requested_load_window"]
    target.pop("requested_load_window", None)
    return None


def read_ollama_window(target, requested):
    """Statement for the next load in explicit mode, or the actual model's loaded buffer in Auto."""
    import httpx
    from ouroboros.llm_ollama import OllamaProgramChanged, OllamaRouteUnsettledError, _client_options, _drive, _load_window, _metadata_reader
    from ouroboros.utils import transport_no_proxy

    try:
        with httpx.Client(**_client_options(target, transport_no_proxy(), READ_TIMEOUT_SEC)) as client:
            selected, loaded = _drive(_load_window(target, requested, sizing_only=True), _metadata_reader(client, target))
            return selected if requested else loaded
    except OllamaProgramChanged:
        raise
    except OllamaRouteUnsettledError as error:
        if not error.retryable:
            raise
        return None  # no observation; an answered but empty /ps is zero


def ollama_model_maximum(shown):
    """Read the maximum for the architecture named in a successful /show."""
    info = shown.get("model_info") if isinstance(shown, dict) else None
    if not isinstance(info, dict):
        return 0
    if "general.architecture" in info:
        return _positive(info.get(f"{info['general.architecture']}.context_length"))
    lengths = {_positive(value) for key, value in info.items() if key.endswith(".context_length")}
    return lengths.pop() if len(lengths) == 1 else 0


def _ollama_window_target(provider, model, base_url, *, api_key=None, headers=None):
    """Use this route's credentials and custom headers, never another connection's."""
    from ouroboros.llm_routing import resolve_target
    from ouroboros.provider_models import parse_model_route
    target = dict(provider=provider, requested_model=model, resolved_model=parse_model_route(model).model, base_url=base_url)
    try:
        resolved = resolve_target(model)
        if resolved.get('provider') == provider and server_root(resolved.get('base_url')) == server_root(base_url):
            target = {**resolved, **target}
    except Exception:
        pass
    if api_key is not None:
        target['api_key'] = api_key
    if headers is not None:
        target['default_headers'] = headers
    return target
