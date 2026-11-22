"""The window an OpenAI-compatible server states about itself, read from its own documented GETs.

Each reader names the server, the documented path and the field (``READERS``, ``MODELS_FIELDS``). The window read
here is the one serving buffer the input and the reply share, per request (llama.cpp's slot ``n_ctx`` is already
divided by ``--parallel``), so it is recorded with source ``server_metadata`` (``context_window._REPLY_INSIDE``).
A gateway catalog's ``context_length`` keeps its older meaning, ``provider_metadata``.

Every failure (refused connection, timeout, non-JSON, another model loaded, a missing or zero field) is ``0``,
"unknown": never a guessed window and never an error the owner sees. Training lengths (llama.cpp ``n_ctx_train``,
Ollama ``/api/show``, LM Studio ``max_context_length``) are what the model could take, not what is loaded, and
are never read as the window.
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


def _ollama_ps(payload: Any, wanted: set) -> int:
    """Ollama ``GET /api/ps``: ``context_length`` the scheduler allocated to the loaded model (0 before a load)."""
    items = payload.get("models") if isinstance(payload, dict) else None
    for item in items if isinstance(items, list) else ():
        if isinstance(item, dict) and (_same_model(item.get("name"), wanted) or _same_model(item.get("model"), wanted)):
            return _positive(item.get("context_length"))
    return 0


#: Readers at the server root (``base_url`` without its ``/v1``), asked in order when ``/models`` stated no window.
READERS: Tuple[Tuple[str, str, Callable[[Any, set], int]], ...] = (
    ("llama.cpp", "/props", _llama_props),
    ("LM Studio", "/api/v0/models", _lmstudio_v0),
    ("Ollama", "/api/ps", _ollama_ps),
)


def server_root(base_url: str) -> str:
    root = str(base_url or "").strip().rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def private_host(base_url: str) -> bool:
    """Loopback, private-network or ``.local`` hosts: the servers the root readers are meant for. A public gateway is
    never asked for another product's paths."""
    import ipaddress
    from urllib.parse import urlsplit

    host = (urlsplit(str(base_url or "")).hostname or "").lower()
    if host in {"localhost"} or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


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



def read_metadata_window(model: str, base_url: str, api_key: str, *, provider: str, evidence_root) -> Tuple[int, str]:
    """CW6 (v6.34.0): an OpenAI-compatible server (vLLM, Ollama, LM Studio, TGI, ...)
    commonly publishes the per-model window in GET {base_url}/models (Go: its documented list URL). vLLM's
    ``max_model_len`` and llama.cpp's ``meta.n_ctx`` are the server's own buffer (``server_metadata``), a catalog's
    ``context_length`` / ``context_window`` stay ``provider_metadata``. When the listing answered without one, a
    server on this host or network is asked at its own documented paths (llama.cpp ``/props``, LM Studio
    ``/api/v0/models``, Ollama ``/api/ps``). Best-effort, fail-closed to ``(0, "")`` (network/auth/parse error,
    no base_url, or hot-path allow_fetch=False).

    The caller supplies the route key and evidence root; this reader never resolves another connection.
    """
    try:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        from ouroboros.net_transport import verify_kwargs
        from ouroboros.opencode_go import compatible_models_url, go_client_headers, go_model_limits, is_go_chat_route

        headers.update(go_client_headers(provider, base_url))
        resp = httpx.get(compatible_models_url(provider, base_url), headers=headers, timeout=5.0, **verify_kwargs())
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

        go_route = is_go_chat_route(provider, base_url)
        maximum = (max((go_model_limits(item)[1] for item in matched), default=0) if go_route else
                   max((positive_limit(item.get("max_output_tokens")) for item in matched), default=0))
        # A listing that was read is an observation even when it publishes no maximum.
        record_metadata_limit(evidence_root, provider=provider, model=model,
            base_url=base_url, maximum=maximum,
            source="OpenCode Go /models limit.output" if go_route else "/models max_output_tokens")
        if go_route:
            window = max((go_model_limits(item)[0] for item in matched), default=0)
            return (window, SOURCE_PROVIDER_METADATA) if window else (0, "")
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
