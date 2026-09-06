"""Provider model-catalog endpoint helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.config import load_settings
from ouroboros.gateway._helpers import json_error, json_exception
from ouroboros.observability import redact_projection
from ouroboros.provider_models import (
    ALL_PROVIDER_CREDENTIAL_KEYS,
    ACTIVE_MODEL_SETTING_KEYS,
    DEEPSEEK_BASE_URL,
    DIRECT_PROVIDER_DEFAULTS,
    MINIMAX_REGION_ENDPOINTS,
    OPENROUTER_DEFAULTS,
    provider_for_model,
    resolve_minimax_base_url,
)

log = logging.getLogger(__name__)

_CATALOG_HTTP_TIMEOUT_SEC = 20.0


def _provider_label_from_model_id(model_id: str) -> str:
    prefix = str(model_id or "").split("/", 1)[0].strip().lower()
    return {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google",
        "meta-llama": "Meta",
        "x-ai": "xAI",
        "qwen": "Qwen",
        "mistralai": "Mistral",
        "deepseek": "DeepSeek",
        "perplexity": "Perplexity",
    }.get(prefix, prefix.title() if prefix else "Other")


def _tagged_model_value(provider_id: str, model_id: str) -> str:
    model_value = str(model_id or "").strip()
    if provider_id == "openrouter":
        return model_value
    return f"{provider_id}::{model_value}"


def _build_model_catalog_entry(
    provider_id: str,
    provider_label: str,
    model_id: str,
    display_name: str,
    source: str | None = None,
) -> dict[str, str]:
    raw_id = str(model_id or "").strip()
    name = str(display_name or "").strip() or raw_id
    source_label = source or provider_label
    return {
        "provider_id": provider_id,
        "provider": provider_label,
        "source": source_label,
        "id": raw_id,
        "name": name,
        "value": _tagged_model_value(provider_id, raw_id),
        "label": f"{source_label} · {name}",
    }


async def _fetch_openrouter_model_catalog(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict[str, str]]:
    response = await client.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    data = response.json()
    raw_models = data.get("data", []) or []

    models: list[dict[str, str]] = []
    for item in raw_models:
        model_id = str(item.get("id", "") or "").strip()
        if not model_id or "/" not in model_id:
            continue
        models.append(
            _build_model_catalog_entry(
                "openrouter",
                _provider_label_from_model_id(model_id),
                model_id,
                str(item.get("name", "") or "").strip() or model_id.split("/", 1)[1],
                source="OpenRouter",
            )
        )
    return models


async def _fetch_openai_compatible_model_catalog(
    client: httpx.AsyncClient,
    provider_id: str,
    provider_label: str,
    api_key: str,
    base_url: str,
) -> list[dict[str, str]]:
    api_root = str(base_url or "").rstrip("/")
    if not api_root:
        return []

    headers = {"Authorization": f"Bearer {api_key}"} if str(api_key or "").strip() else None
    response = await client.get(
        f"{api_root}/models",
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    raw_models = data.get("data", []) or []

    models: list[dict[str, str]] = []
    for item in raw_models:
        model_id = str(item.get("id", "") or "").strip()
        if not model_id:
            continue
        models.append(
            _build_model_catalog_entry(
                provider_id,
                provider_label,
                model_id,
                str(item.get("name", "") or "").strip() or model_id,
            )
        )
    return models


async def _fetch_anthropic_model_catalog(
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict[str, str]]:
    response = await client.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    response.raise_for_status()
    data = response.json()
    raw_models = data.get("data", []) or []

    models: list[dict[str, str]] = []
    for item in raw_models:
        model_id = str(item.get("id", "") or "").strip()
        if not model_id:
            continue
        models.append(
            _build_model_catalog_entry(
                "anthropic",
                "Anthropic",
                model_id,
                str(item.get("display_name", "") or item.get("name", "") or "").strip() or model_id,
            )
        )
    return models


async def _fetch_gigachat_model_catalog(
    credentials: str,
    scope: str,
    base_url: str,
    verify_ssl_certs: bool,
    user: str = "",
    password: str = "",
) -> list[dict[str, str]]:
    """List GigaChat models via the `gigachat` library (not OpenAI-compatible).

    Auth is either an authorization key (OAuth) or user/password. Uses the
    library's native async client so no blocking call / thread offload is needed
    (the catalog loader stays fully async and fork-safe). Failures propagate to
    the loader, which degrades gracefully per-provider.
    """
    from gigachat import GigaChatAsyncClient

    kwargs: dict = {"scope": scope or "GIGACHAT_API_PERS", "verify_ssl_certs": verify_ssl_certs}
    if credentials:
        kwargs["credentials"] = credentials
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password
    if base_url:
        kwargs["base_url"] = base_url

    async with GigaChatAsyncClient(**kwargs) as client:
        result = await client.aget_models()

    entries: list[dict[str, str]] = []
    for model in getattr(result, "data", None) or []:
        model_id = str(getattr(model, "id_", "") or "").strip()
        if not model_id:
            continue
        entries.append(
            _build_model_catalog_entry("gigachat", "GigaChat", model_id, model_id)
        )
    return entries


def _provider_specs(
    settings: dict,
) -> list[tuple[str, Callable[[httpx.AsyncClient], Awaitable[list[dict[str, str]]]]]]:
    specs: list[tuple[str, Callable[[httpx.AsyncClient], Awaitable[list[dict[str, str]]]]]] = []

    openrouter_api_key = str(settings.get("OPENROUTER_API_KEY", "") or "").strip()
    if openrouter_api_key:
        specs.append(("openrouter", lambda client: _fetch_openrouter_model_catalog(client, openrouter_api_key)))

    openai_api_key = str(settings.get("OPENAI_API_KEY", "") or "").strip()
    if openai_api_key:
        specs.append((
            "openai",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "openai",
                "OpenAI",
                openai_api_key,
                "https://api.openai.com/v1",
            ),
        ))

    anthropic_api_key = str(settings.get("ANTHROPIC_API_KEY", "") or "").strip()
    if anthropic_api_key:
        specs.append(("anthropic", lambda client: _fetch_anthropic_model_catalog(client, anthropic_api_key)))

    minimax_api_key = str(settings.get("MINIMAX_API_KEY", "") or "").strip()
    if minimax_api_key:
        # MiniMax serves an OpenAI-compatible GET /v1/models on the region host
        # (platform.minimax.io API reference), so the catalog is fetched live like
        # every other remote provider instead of shipping a hardcoded list.
        minimax_base_url = resolve_minimax_base_url(
            str(settings.get("MINIMAX_REGION", "") or "")
        )
        specs.append((
            "minimax",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "minimax",
                "MiniMax",
                minimax_api_key,
                minimax_base_url,
            ),
        ))

    deepseek_api_key = str(settings.get("DEEPSEEK_API_KEY", "") or "").strip()
    if deepseek_api_key:
        # DeepSeek serves an OpenAI-compatible GET /models on its one official
        # host, so the catalog is fetched live like the other remote providers.
        specs.append((
            "deepseek",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "deepseek",
                "DeepSeek",
                deepseek_api_key,
                DEEPSEEK_BASE_URL,
            ),
        ))

    compatible_api_key = str(settings.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
    compatible_base_url = str(settings.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip()
    legacy_base_url = str(settings.get("OPENAI_BASE_URL", "") or "").strip()
    if compatible_base_url:
        specs.append((
            "openai-compatible",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "openai-compatible",
                "OpenAI Compatible",
                compatible_api_key,
                compatible_base_url,
            ),
        ))
    elif legacy_base_url:
        specs.append((
            "openai-compatible",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "openai-compatible",
                "OpenAI Compatible",
                compatible_api_key or openai_api_key,
                legacy_base_url,
            ),
        ))

    cloudru_api_key = str(settings.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip()
    if cloudru_api_key:
        cloudru_base_url = str(settings.get("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "") or "").strip()
        if not cloudru_base_url:
            cloudru_base_url = "https://foundation-models.api.cloud.ru/v1"
        specs.append((
            "cloudru",
            lambda client: _fetch_openai_compatible_model_catalog(
                client,
                "cloudru",
                "Cloud.ru",
                cloudru_api_key,
                cloudru_base_url,
            ),
        ))

    gigachat_credentials = str(settings.get("GIGACHAT_CREDENTIALS", "") or "").strip()
    gigachat_user = str(settings.get("GIGACHAT_USER", "") or "").strip()
    gigachat_password = str(settings.get("GIGACHAT_PASSWORD", "") or "").strip()
    if gigachat_credentials or (gigachat_user and gigachat_password):
        gigachat_scope = str(settings.get("GIGACHAT_SCOPE", "") or "").strip() or "GIGACHAT_API_PERS"
        gigachat_base_url = str(settings.get("GIGACHAT_BASE_URL", "") or "").strip()
        gigachat_verify = str(settings.get("GIGACHAT_VERIFY_SSL_CERTS", "true") or "").strip().lower()
        gigachat_verify_bool = gigachat_verify not in ("0", "false", "no", "off")
        # GigaChat isn't OpenAI-compatible; the loader ignores the shared httpx
        # client and uses the gigachat library directly (auth via key or basic).
        specs.append((
            "gigachat",
            lambda _client: _fetch_gigachat_model_catalog(
                gigachat_credentials,
                gigachat_scope,
                gigachat_base_url,
                gigachat_verify_bool,
                gigachat_user,
                gigachat_password,
            ),
        ))

    return specs


def _catalog_error_stage(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "connect"
    if isinstance(exc, httpx.HTTPStatusError):
        return "http"
    if isinstance(exc, httpx.TransportError):
        return "transport"
    if isinstance(exc, ValueError):
        return "parse"
    return "error"


async def _load_provider(
    client: httpx.AsyncClient,
    provider_id: str,
    loader: Callable[[httpx.AsyncClient], Awaitable[list[dict[str, str]]]],
) -> tuple[str, list[dict[str, str]], str, str, int]:
    started = time.perf_counter()
    log.info("model_catalog provider=%s stage=start", provider_id)
    try:
        items = await loader(client)
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "model_catalog provider=%s stage=success duration_ms=%s item_count=%s",
            provider_id,
            duration_ms,
            len(items),
        )
        return provider_id, items, "", "", duration_ms
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        stage = _catalog_error_stage(exc)
        log.warning(
            "model_catalog provider=%s stage=%s duration_ms=%s error=%s",
            provider_id,
            stage,
            duration_ms,
            exc,
        )
        return provider_id, [], str(exc), stage, duration_ms


async def api_model_catalog(_request: Request) -> JSONResponse:
    settings = load_settings()
    items: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    seen_values: set[str] = set()
    specs = _provider_specs(settings)

    timeout = httpx.Timeout(_CATALOG_HTTP_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(*[
            _load_provider(client, provider_id, loader)
            for provider_id, loader in specs
        ])

    for provider_id, provider_items, error, stage, duration_ms in results:
        if error:
            errors.append({
                "provider_id": provider_id,
                "error": error,
                "stage": stage,
                "duration_ms": duration_ms,
            })
            continue
        for item in provider_items:
            value = str(item.get("value", "") or "")
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            items.append(item)

    items.sort(key=lambda item: (item.get("provider", "").lower(), item.get("name", "").lower()))
    return JSONResponse({
        "items": items,
        "errors": errors,
    })


async def api_local_model_start(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        source = body.get("source", "").strip()
        filename = body.get("filename", "").strip()
        port = int(body.get("port", 8766))
        n_gpu_layers = int(body.get("n_gpu_layers", -1))
        n_ctx = int(body.get("n_ctx", 0))
        chat_format = body.get("chat_format", "").strip()

        if not source:
            return JSONResponse({"error": "source is required"}, status_code=400)

        from ouroboros.local_model import get_manager, _get_runtime_hint
        mgr = get_manager()

        if mgr.is_running:
            return JSONResponse({"error": "Local model server is already running"}, status_code=409)

        # Preflight: check llama-cpp-python is installed BEFORE downloading the model.
        # This prevents users from waiting through a large download only to hit an
        # install error at the end.
        # Run in a thread to avoid blocking the async event loop (subprocess.run, 15s timeout).
        runtime_ok = await asyncio.to_thread(mgr.check_runtime)
        if not runtime_ok:
            return JSONResponse(
                {
                    "error": "runtime_missing",
                    "message": (
                        "llama-cpp-python is not installed. "
                        "Use the 'Install Local Runtime' button to install it first."
                    ),
                    "hint": _get_runtime_hint(),
                },
                status_code=412,
            )

        # Download can be slow, run in thread to not block the async event loop
        model_path = await asyncio.to_thread(mgr.download_model, source, filename)

        mgr.start_server(model_path, port=port, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, chat_format=chat_format)
        return JSONResponse({"status": "starting", "model_path": model_path})
    except Exception as e:
        return json_exception(e)


async def api_local_model_stop(request: Request) -> JSONResponse:
    try:
        from ouroboros.local_model import get_manager
        get_manager().stop_server()
        return JSONResponse({"status": "stopped"})
    except Exception as e:
        return json_exception(e)


async def api_local_model_status(request: Request) -> JSONResponse:
    try:
        from ouroboros.local_model import get_manager
        mgr = get_manager()
        # If runtime status is still unknown and no operation is running,
        # run a quick probe so the Settings page can surface the Install button
        # on the very first poll — before the user clicks Start.
        if mgr._runtime_status == "unknown" and mgr.get_status() == "offline":
            await asyncio.to_thread(mgr.check_runtime)
        return JSONResponse(mgr.status_dict())
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


async def api_local_model_test(request: Request) -> JSONResponse:
    try:
        from ouroboros.local_model import get_manager
        mgr = get_manager()
        if not mgr.is_running:
            return json_error("Local model server is not running", 400)
        result = mgr.test_tool_calling()
        return JSONResponse(result)
    except Exception as e:
        return json_exception(e)


async def api_openai_compatible_models(request: Request) -> JSONResponse:
    """Proxy GET {baseUrl}/models so the onboarding wizard avoids browser CORS limits."""
    try:
        body = await request.json()
        base_url = str(body.get("baseUrl", "") or "").strip()
        api_key = str(body.get("apiKey", "") or "").strip()
        if not base_url:
            return json_error("baseUrl is required", 400)
        async with httpx.AsyncClient(timeout=10.0) as client:
            models = await _fetch_openai_compatible_model_catalog(
                client, "openai-compatible", "OpenAI-compatible", api_key, base_url
            )
        model_ids = [m["value"].removeprefix("openai-compatible::") for m in models if m.get("value")]
        return JSONResponse({"models": model_ids})
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": f"HTTP {e.response.status_code}"}, status_code=502)
    except Exception as e:
        return json_exception(e)


_PROVIDER_TEST_OVERRIDE_KEYS = ALL_PROVIDER_CREDENTIAL_KEYS
_PROVIDER_TEST_KNOWN_IDS: frozenset[str] = frozenset({
    "openrouter", "openai-compatible", *DIRECT_PROVIDER_DEFAULTS,
})


def _provider_test_model(provider_id: str, settings: dict) -> str:
    """Resolve one deterministic runtime/default model without catalog I/O."""
    for setting_key in ACTIVE_MODEL_SETTING_KEYS:
        for raw_model in str(settings.get(setting_key, "") or "").split(","):
            model = raw_model.strip()
            if model and provider_for_model(model) == provider_id:
                return model
    if provider_id == "openrouter":
        return str(OPENROUTER_DEFAULTS.get("main") or "").strip()
    return str((DIRECT_PROVIDER_DEFAULTS.get(provider_id) or {}).get("main") or "").strip()


def _discover_provider_test_model(provider_id: str, settings: dict) -> str:
    """Catalog discovery for the sole route without a maintained default."""
    if provider_id != "openai-compatible":
        return ""

    async def load_first() -> str:
        loader = dict(_provider_specs(settings)).get(provider_id)
        if loader is None:
            return ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(_CATALOG_HTTP_TIMEOUT_SEC)) as client:
            items = await loader(client)
        return str((items[0] if items else {}).get("value") or "").strip()

    return asyncio.run(load_first())


def _provider_test_log(provider_id: str, model: str, result: dict, duration_ms: int) -> None:
    facts = redact_projection({
        "provider_id": provider_id,
        "model": model,
        "ok": bool(result.get("ok")),
        "error_category": str(result.get("error") or ""),
        "status_code": result.get("status_code"),
        "exception_type": str(result.get("exception_type") or ""),
        "duration_ms": duration_ms,
    }).value
    logger = log.info if facts.get("ok") else log.warning
    logger("provider_test result=%s", facts)


def _run_provider_test_with_settings(provider_id: str, settings: dict) -> dict:
    """Run selection, optional discovery, client construction and one send."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_probe import controlled_probe_error

    started = time.perf_counter()
    model = ""
    try:
        model = _provider_test_model(provider_id, settings)
        if not model:
            model = _discover_provider_test_model(provider_id, settings)
        if not model:
            result = {
                "ok": False,
                "error": "Provider is not configured",
                "status_code": None,
                "exception_type": "ProviderNotConfigured",
            }
        else:
            result = LLMClient().probe_provider_readiness(model, settings=settings)
    except Exception as exc:
        result = controlled_probe_error(exc)
    duration_ms = int((time.perf_counter() - started) * 1000)
    _provider_test_log(provider_id, model, result, duration_ms)
    public = {"ok": bool(result.get("ok"))}
    if not public["ok"]:
        public["error"] = str(result.get("error") or "Model request failed")
    return public


def _run_provider_test(provider_id: str, overrides: dict[str, str]) -> dict:
    """Load and merge the request-local draft inside the worker thread."""
    settings = dict(load_settings())
    if provider_id == "openai-compatible":
        # No-edit requests retain the historical OPENAI_* fallback. An explicit
        # Clear on the dedicated card must not silently restore the corresponding
        # legacy value for this request-local probe.
        legacy_fallbacks = {
            "OPENAI_COMPATIBLE_API_KEY": "OPENAI_API_KEY",
            "OPENAI_COMPATIBLE_BASE_URL": "OPENAI_BASE_URL",
        }
        if not any(str(settings.get(key, "") or "").strip() for key in legacy_fallbacks):
            # A legacy install has no dedicated compatible pair yet. Preserve the
            # saved half the owner did not edit while the other half is a draft.
            for dedicated_key, legacy_key in legacy_fallbacks.items():
                if dedicated_key not in overrides:
                    settings[dedicated_key] = str(settings.get(legacy_key, "") or "").strip()
        for dedicated_key, legacy_key in legacy_fallbacks.items():
            if dedicated_key in overrides and not overrides[dedicated_key].strip():
                settings[legacy_key] = ""
    settings.update({key: value.strip() for key, value in overrides.items()})
    minimax_region = str(settings.get("MINIMAX_REGION", "") or "").strip().lower()
    if provider_id == "minimax" and minimax_region and minimax_region not in MINIMAX_REGION_ENDPOINTS:
        return {"error": "unknown MiniMax region", "_http_status": 400}
    return _run_provider_test_with_settings(provider_id, settings)


async def api_provider_test(request: Request) -> JSONResponse:
    """Run one request-local, physically accounted completion probe."""
    try:
        body = await request.json()
    except Exception:
        return json_error("request body must be a JSON object", 400)
    if not isinstance(body, dict):
        return json_error("request body must be a JSON object", 400)
    # Executable gateway ABI (ABI-3, Q7=A): ProviderTestRequest schema gate.
    from ouroboros.gateway.contracts import ProviderTestRequest
    from ouroboros.gateway.schema import validate_ingress

    schema_errors = validate_ingress(body, ProviderTestRequest)
    if schema_errors:
        return json_error(
            f"invalid request body: {schema_errors[0]}", 400,
            schema_errors=schema_errors[:8],
        )
    provider_id = str(body.get("provider_id", "") or "").strip()
    if not provider_id:
        return json_error("provider_id is required", 400)
    if provider_id not in _PROVIDER_TEST_KNOWN_IDS:
        return json_error("unknown provider id", 400)

    overrides = body.get("overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        return json_error("overrides must be an object", 400)
    unknown = sorted(str(key) for key in overrides if str(key) not in _PROVIDER_TEST_OVERRIDE_KEYS)
    if unknown:
        return json_error(f"unsupported override keys: {', '.join(unknown)}", 400)
    if any(not isinstance(value, str) for value in overrides.values()):
        return json_error("override values must be strings", 400)

    result = await asyncio.to_thread(
        _run_provider_test,
        provider_id,
        {str(key): value for key, value in overrides.items()},
    )
    status_code = int(result.pop("_http_status", 200))
    return JSONResponse(result, status_code=status_code)


async def api_local_model_install_runtime(request: Request) -> JSONResponse:
    """Start an async install of llama-cpp-python into the app-managed interpreter.

    The install runs in a background thread tracked on the manager.  Callers
    should poll ``/api/local-model/status`` and watch ``runtime_status``:

    - ``"installing"``   — install in progress
    - ``"install_ok"``   — install succeeded; caller may now start the model
    - ``"install_error"``— install failed; ``runtime_install_log`` has details
    """
    try:
        from ouroboros.local_model import get_manager
        mgr = get_manager()

        current = mgr._runtime_status
        if current == "installing":
            return JSONResponse({"status": "already_installing"})

        mgr.install_runtime()
        return JSONResponse({"status": "installing"})
    except Exception as e:
        return json_exception(e)
