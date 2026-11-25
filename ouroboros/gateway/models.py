"""Provider catalog discovery, shared by saved pickers and request-local drafts.

Only saved discovery opts into image evidence for compatible, Anthropic and
DeepSeek catalogs. Record their already-fetched rows off the event loop through
the existing evidence owner, before reducing them to the public picker shape.
Draft onboarding and Provider Test never opt in, even with saved credentials.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from typing import TYPE_CHECKING, Awaitable, Callable

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.config import load_settings
from ouroboros.llm_pricing import openrouter_catalog_price
from ouroboros.net_transport import ExtraCaBundleError, extra_ca_bundle, verify_kwargs
from ouroboros.gateway._helpers import json_error, json_exception, run_sync_to_completion
from ouroboros.observability import redact_projection
from ouroboros.provider_connections import (
    connection_configured, connection_display, connection_routes, connection_rows, direct_provider_defaults, find_row,
    seed_new_api_surfaces,
)
from ouroboros.provider_models import (
    ACTIVE_MODEL_SETTING_KEYS,
    BUILTIN_CONNECTION,
    COMPATIBLE_LEGACY_PAIR,
    CONNECTIONS_KEY,
    DIRECT_PROVIDER_DEFAULTS,
    OPENROUTER_DEFAULTS,
    PROVIDER_KINDS,
    choice_error,
    compatible_endpoint_configured,
    connection_id,
    parse_model_route,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ouroboros.gateway.management_contracts import ModelCatalogApiSource

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
        "z-ai": "Z.ai (GLM)",
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
    cache_ttl: str = "",
) -> list[dict]:
    response = await client.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    data = response.json()
    raw_models = data.get("data", []) or []

    models: list[dict] = []
    for item in raw_models:
        model_id = str(item.get("id", "") or "").strip()
        if not model_id or "/" not in model_id:
            continue
        entry = _build_model_catalog_entry(
            "openrouter",
            _provider_label_from_model_id(model_id),
            model_id,
            str(item.get("name", "") or "").strip() or model_id.split("/", 1)[1],
            source="OpenRouter",
        )
        price = openrouter_catalog_price(item, cache_ttl)  # only this source's rows: a direct key's same id stays unpriced
        # The provider's published window, for display beside the price: sizing reads its own evidence.
        window = item.get("context_length")
        if type(window) is int and window > 0:
            entry["context_window"] = window
        models.append(entry if price is None else {**entry, "price": price})
    return models


async def _fetch_openai_compatible_model_catalog(
    client: httpx.AsyncClient,
    provider_id: str,
    provider_label: str,
    api_key: str,
    base_url: str,
    *,
    record_image_input: bool = False,
) -> list[dict[str, str]]:
    from ouroboros.opencode_go import compatible_models_url, go_client_headers

    api_root = str(base_url or "").rstrip("/")
    if not api_root:
        return []

    headers = go_client_headers(provider_id, base_url)
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key}"
    response = await client.get(
        compatible_models_url(provider_id, base_url),
        headers=headers or None,
    )
    response.raise_for_status()
    data = response.json()
    raw_models = data.get("data", []) or []

    if record_image_input:
        from ouroboros.vision_routing import record_catalog_image_input

        await asyncio.to_thread(record_catalog_image_input, provider_id, api_root, raw_models,
                                source=f"the {provider_label} /models catalog", api_key=api_key if "Authorization" in headers else "")

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
    if provider_id == "deepseek":  # what its published entry prices only under a condition, never a figure
        from ouroboros.pricing_catalog import price_condition

        conditions = await asyncio.to_thread(lambda: [price_condition(entry["id"]) for entry in models])
        models = [{**entry, "price_condition": said} if said else entry for entry, said in zip(models, conditions)]
    return models


async def _fetch_anthropic_model_catalog(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    record_image_input: bool = False,
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

    if record_image_input:
        from ouroboros.vision_routing import record_catalog_image_input

        await asyncio.to_thread(record_catalog_image_input, "anthropic", "https://api.anthropic.com/v1",
                                raw_models, source="the Anthropic /models catalog", api_key=api_key)

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
    bundle = extra_ca_bundle()
    if bundle:
        kwargs["ca_bundle_file"] = bundle
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


# The catalog label of each kind's built-in items (OpenRouter items carry their vendor).
_CATALOG_LABELS = {"openai": "OpenAI", "minimax": "MiniMax", "deepseek": "DeepSeek", "zai": "Z.ai (GLM)",
                   "openai-compatible": "OpenAI Compatible", "cloudru": "Cloud.ru"}


def _connection_loader(kind: str, target: dict, record_image_input: bool, cache_ttl: str = ""):
    """The live ``/models`` read of one connection, from its resolved target (key and endpoint)."""
    key, base = str(target.get("api_key") or "").strip(), str(target.get("base_url") or "")
    if kind == "openrouter":
        return lambda client: _fetch_openrouter_model_catalog(client, key, cache_ttl)
    if kind == "anthropic":
        return lambda client: _fetch_anthropic_model_catalog(client, key, record_image_input=record_image_input)
    if kind == "gigachat":
        # GigaChat isn't OpenAI-compatible; the loader ignores the shared httpx
        # client and uses the gigachat library directly (auth via key or basic).
        return lambda _client: _fetch_gigachat_model_catalog(
            key, target["scope"], base, target["verify_ssl_certs"], target["user"], target["password"])
    # These routes share the OpenAI list shape; each request uses the connection's own key.
    record = {"record_image_input": True} if record_image_input and kind in {"deepseek", "openai-compatible"} else {}
    return lambda client: _fetch_openai_compatible_model_catalog(client, kind, _CATALOG_LABELS[kind], key, base, **record)


def _labelled_loader(loader, source: ModelCatalogApiSource, *, named: bool):
    """Items of a named connection carry its route spelling and label, so the same model
    on two keys stays two entries; built-in items keep their values (all: ``transport``)."""
    async def load(client):
        items = await loader(client)
        if "processing_transport" in source:  # Compatibility for clients that only know item rows.
            items = [{**item, "processing_transport": source["processing_transport"]} for item in items]
        if not named:
            return items
        source_id, display = source["id"], source["label"]
        return [{**item, "provider_id": source_id, "provider": source_id, "provider_label": display, "source": display,
                 "value": f"{source_id}::{item['id']}", "label": f"{display} · {item['name']}"} for item in items]
    return load


def _provider_specs(
    settings: dict,
    *,
    record_image_input: bool = False,
    provider_id: str | None = None,
) -> list[tuple[ModelCatalogApiSource, Callable[[httpx.AsyncClient], Awaitable[list[dict]]]]]:
    """One loader per configured connection, with the resolved host transport alongside, not inside its read."""
    from ouroboros.llm_attempt import ConnectionUnavailable, processing_transport_modes
    from ouroboros.llm_routing import resolve_target

    rows = connection_rows(settings.get(CONNECTIONS_KEY))
    from ouroboros.settings_scales import resolve_prompt_cache_ttl

    cache_ttl = resolve_prompt_cache_ttl(settings.get("OUROBOROS_PROMPT_CACHE_TTL") or "")  # the tier a row's cache-write figure is of
    specs = []
    for route in connection_routes(rows):
        kind, name = route.kind, route.connection
        source = connection_id(kind, name)
        if provider_id is not None and provider_id != source:
            continue
        try:
            target = resolve_target(f"{source}::catalog", settings)
        except ConnectionUnavailable:
            continue
        # A compatible endpoint lists its models with or without a key (the legacy base too).
        if target.get("base_url") if kind == "openai-compatible" else connection_configured(route, settings.get, rows):
            modes = processing_transport_modes(target)
            descriptor: ModelCatalogApiSource = {"id": source, "label": connection_display(kind, name, rows)}
            if modes is not None:
                descriptor["processing_transport"] = list(modes)
            specs.append((descriptor, _labelled_loader(_connection_loader(kind, target, record_image_input, cache_ttl),
                                                  descriptor, named=bool(name))))
    return specs


def _local_model_entry(settings: dict) -> dict | None:
    """The configured local model as one row with ``use_local``, or None when none is configured.

    A local model has no route prefix (a "use local" flag reaches it), so ``id`` names the artifact and
    ``value`` is a label no API source claims, never a route to store."""
    source = str(settings.get("LOCAL_MODEL_SOURCE") or "").strip()
    if not source:
        return None
    filename = str(settings.get("LOCAL_MODEL_FILENAME") or "").strip()
    name = (filename or source).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or source
    window = settings.get("LOCAL_MODEL_CONTEXT_LENGTH")
    known = {"context_window": window} if type(window) is int and window > 0 else {}
    from ouroboros.llm_attempt import processing_transport_modes
    return {**_build_model_catalog_entry("local", "Local", name, name), "use_local": True,
            "processing_transport": list(processing_transport_modes({"provider": "local"})), **known}


def _catalog_error_stage(exc: Exception) -> str:
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
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
) -> tuple[list[dict[str, str]], dict | None]:
    """One connection's rows, or its error row for the catalog's ``errors``."""
    started = time.perf_counter()
    log.info("model_catalog provider=%s stage=start", provider_id)
    try:
        items = await asyncio.wait_for(loader(client), timeout=_CATALOG_HTTP_TIMEOUT_SEC)
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "model_catalog provider=%s stage=success duration_ms=%s item_count=%s",
            provider_id,
            duration_ms,
            len(items),
        )
        return items, None
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        stage = _catalog_error_stage(exc)
        message = str(exc) or (f"Catalog read exceeded {_CATALOG_HTTP_TIMEOUT_SEC:g}s" if stage == "timeout" else type(exc).__name__)
        log.warning(
            "model_catalog provider=%s stage=%s duration_ms=%s error=%s",
            provider_id,
            stage,
            duration_ms,
            message,
        )
        return [], {"provider_id": provider_id, "error": message, "stage": stage, "duration_ms": duration_ms}


def account_catalog_supported(operations: list[dict], path: str) -> bool:
    """Opt in only when this exact operation declares the accounts query view."""
    return any(
        operation.get("method") == "GET" and operation.get("path") == path
        and any(parameter.get("name") == "view" and parameter.get("location") == "query"
                and "accounts" in (parameter.get("enum") or [])
                for parameter in operation.get("parameters", []) if isinstance(parameter, dict))
        for operation in operations if isinstance(operation, dict)
    )


def account_catalog_models(envelope: dict):
    """Keep each model attached to the account and original catalog that reported it."""
    for account in envelope.get("accounts", []):
        catalog = account.get("catalog")
        if isinstance(catalog, dict):
            for model in catalog.get("models", []):
                if isinstance(model, dict):
                    yield account, catalog, model


# The latest catalog observation of a selected profile. A save may compare it and the
# current engine store without fetching provider models or generating a request.
_profile_lock = threading.Lock()
_profile_reads: dict[str, tuple[int, dict]] = {}


def _remember_profiles(source: str, pin: str, catalog: dict, store: str, revision: int, started: int):
    rows = [catalog]
    if "accounts" in catalog:
        rows = [row["catalog"] for row in catalog.get("accounts") or [] if isinstance(row, dict)
                and isinstance(row.get("catalog"), dict)
                and row["catalog"].get("credentialProfileId") == row.get("credentialProfileId")]
    identities = {row["credentialProfileId"]: (store, row["accountFingerprint"], revision) for row in rows
                  if store and row.get("source") == source and isinstance(row.get("credentialProfileId"), str)
                  and isinstance(row.get("accountFingerprint"), str) and row["accountFingerprint"]}
    with _profile_lock:
        full_read, profiles = _profile_reads.get(source, (0, {}))
        if started < full_read:
            return
        if pin:
            if started >= profiles.get(pin, (0, None))[0]:
                profiles[pin] = started, identities.get(pin)
        else:
            # A slower full listing replaces only observations that preceded it.
            profiles = {**{key: (started, identity) for key, identity in identities.items()},
                        **{key: read for key, read in profiles.items() if read[0] > started}}
            full_read = started
        _profile_reads[source] = full_read, profiles


def observed_profile(source: str, pin: str) -> tuple | None:
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.gateway.claudexor_accounts import account_read_revision

    try:
        with read_owned_gateway(timeout_sec=2) as gateway:
            store = gateway.store_identity
    except Exception:
        return None
    with _profile_lock:
        _, observed = _profile_reads.get(source, (0, {}))[1].get(pin, (0, None))
        if observed and store and observed[0] == store and observed[2] == account_read_revision():
            return observed[:2]
    return None


def _one_subscription_catalog(gateway, source: dict, account_view: bool, profile_id: str, *,
                              observe: Callable[[dict], None]) -> dict:
    identifier = source["id"]
    result = {"items": [], "errors": []}
    if account_view:
        result.update(account_catalogs=[], partial=False)
    catalog = gateway.list_source_models(identifier, credential_profile_id=profile_id or None,
                                         **({"view": "accounts"} if account_view else {}))
    observe(catalog if catalog.get("source") == identifier else {})
    if account_view:
        result["account_catalogs"].append(catalog)
        result["partial"] = catalog.get("partial", False)
        for account in catalog.get("accounts", []):
            # A readable account carrying a problem (spent window, cooldown) is account state,
            # already on its items and account_catalogs; only a missing catalog is a read failure.
            if account.get("catalog") is None:
                problem = account.get("problem") or {}
                result["errors"].append({"provider_id": "claudexor", "source_id": identifier,
                    "credential_profile_id": account.get("credentialProfileId"),
                    "code": problem.get("code", "catalog_unavailable"),
                    "error": problem.get("message", "This account catalog could not be read."),
                    "availability": account.get("availability"), "problem": account.get("problem")})
        records = account_catalog_models(catalog)
    else:
        records = ((None, catalog, model) for model in catalog.get("models", []))
    for account, catalog, model in records:
        model_id = str(model.get("id") or "")
        if not model_id:
            continue
        entry = _build_model_catalog_entry("claudexor", str(source.get("label") or identifier),
            model_id, str(model.get("label") or model_id), source="Claudexor")
        entry.update({
            "source_id": identifier,
            **({"sourceProgram": catalog.get("sourceProgram")} if not account_view else {}),
            "value": f"claudexor::{identifier}={model_id}",
            "credential_profile_id": catalog.get("credentialProfileId"),
            "account_fingerprint": catalog.get("accountFingerprint"),
            "context_window": model.get("contextWindow"),
            "max_context_window": model.get("maxContextWindow"),
            "max_output_tokens": model.get("maxOutputTokens"),
            "is_default": model.get("isDefault", False),
            "supported_options": model.get("supportedOptions", []),
            "reasoning_efforts": model.get("reasoningEfforts", []),
            "default_reasoning_effort": model.get("defaultReasoningEffort"),
            "input_modalities": model.get("inputModalities", []),
            "provenance": catalog.get("provenance"),
            "observed_at": catalog.get("observedAt"),
        })
        if account is not None:
            entry.update(availability=account.get("availability"), problem=account.get("problem"),
                         processing=model.get("processing"))
        elif "processing" in model:  # Older catalogs also report per-model tiers, without account proof.
            entry["processing"] = model["processing"]
        result["items"].append(entry)
    return result


def _join_subscription_catalogs(discovery: dict, parts) -> dict:
    result = {**discovery, "items": [], "errors": list(discovery["errors"])}
    if "account_catalogs" in discovery:
        result["account_catalogs"] = []
    for part in parts:
        result["items"].extend(part["items"])
        result["errors"].extend(part["errors"])
        if "account_catalogs" in result:
            result["account_catalogs"].extend(part.get("account_catalogs", []))
        if part.get("partial"):
            result["partial"] = True
    return result


def _subscription_timeout(source=None):
    error = {"provider_id": "claudexor", "stage": "timeout", "error": "Subscription catalogs did not answer in time."}
    if source is not None:
        error["source_id"] = source
    return {"items": [], "errors": [error], "partial": True}


class _SubscriptionBatch:
    """A response samples finished sources at one deadline. The worker owns the gateway
    until its last read finishes, including reads that outlive the HTTP response."""

    def __init__(self):
        self.deadline = time.monotonic() + _CATALOG_HTTP_TIMEOUT_SEC
        self.started = Future()

    def read(self, discovery, sources, read):
        # Discovery itself may have outlived the batch. Do not start new catalog calls for it.
        if not sources:
            return discovery
        if time.monotonic() >= self.deadline:
            return _join_subscription_catalogs(discovery, (_subscription_timeout(s["id"]) for s in sources))
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            jobs = {source["id"]: pool.submit(copy_context().run, read, source) for source in sources}
            self.started.set_result((discovery, jobs))
        return self.snapshot()

    def snapshot(self):
        if not self.started.done():
            return {"model_sources": [], **_subscription_timeout()}
        discovery, jobs = self.started.result()
        return _join_subscription_catalogs(discovery, (
            job.result() if job.done() else _subscription_timeout(source) for source, job in jobs.items()))


def _subscription_model_catalog(source_id: str = "", profile_id: str = "", batch: _SubscriptionBatch | None = None, *, sources_only: bool = False) -> dict:
    """Read raw transport catalogs from the owned engine, never CLI model inventory.

    Account-view engines enumerate every enabled account, retaining independent
    capabilities and failures. Older engines keep selected-account discovery.
    API-only installs do not provision a daemon while browsing model settings.
    """
    from ouroboros.claudexor_daemon import read_owned_gateway, owned_daemon_provisioned

    from ouroboros.gateway.claudexor_accounts import account_read_revision

    revision, started = account_read_revision(), time.monotonic_ns()
    result = {"items": [], "errors": [], "model_sources": []}
    def forget_missing_sources(available):
        with _profile_lock:
            missing = ({source_id} if source_id else set(_profile_reads)) - available
        for source in missing:
            _remember_profiles(source, profile_id, {}, "", revision, started)

    if not source_id and not owned_daemon_provisioned():
        forget_missing_sources(set())
        return result
    try:
        with read_owned_gateway() as gateway:
            try:
                operations = gateway.operations()
            except Exception:
                operations = []  # Discovery compatibility never invents new query support.
            account_view = account_catalog_supported(operations, "/v2/model-sources/:id/models")
            source_view = account_catalog_supported(operations, "/v2/model-sources")
            if isinstance(store := getattr(gateway, "store_identity", ""), str) and store.strip():
                result["store_id"] = store.strip()  # this handshake's store, for this endpoint generation only
            sources = gateway.list_model_sources(**({"view": "accounts"} if source_view else {})).get("sources", [])
            result["model_sources"] = sources
            if sources_only:
                return result
            if account_view:
                result.update(account_catalogs=[], partial=False)
            forget_missing_sources({row.get("id") for row in sources})

            def read(source):
                identifier = source["id"]
                try:
                    return _one_subscription_catalog(
                        gateway, source, account_view, profile_id,
                        observe=lambda catalog: _remember_profiles(
                            identifier, profile_id, catalog, result.get("store_id", ""), revision, started),
                    )
                except Exception as exc:
                    _remember_profiles(identifier, profile_id, {}, "", revision, started)
                    return {"items": [], "partial": account_view, "errors": [{
                        "provider_id": "claudexor", "source_id": source["id"],
                        "code": str(getattr(exc, "code", "catalog_unavailable")), "error": str(exc)}]}

            if batch is not None:
                return batch.read(result, sources, read)
            selected = [s for s in sources if s.get("id") == source_id] if source_id else sources
            if source_id and not selected:
                result["errors"].append({"provider_id": "claudexor", "source_id": source_id,
                    "code": "model_source_unavailable", "error": "This engine does not provide the selected model transport."})
            return _join_subscription_catalogs(result, map(read, selected))
    except Exception as exc:
        forget_missing_sources(set())
        result["errors"].append({"provider_id": "claudexor", "source_id": source_id,
            "code": str(getattr(exc, "code", "catalog_unavailable")), "error": str(exc)})
    return result


async def api_model_catalog(_request: Request) -> JSONResponse:
    query = _request.query_params if _request is not None else {}
    if "processing_model" in query:  # one route's tiers (``pricing.route_processing``), never the catalogs
        if "source_id" in query or "credential_profile_id" in query:
            return json_error("processing_model takes no source_id or credential_profile_id", 400)
        if any(key in query for key in ("view", "provider_id", "harness_id")):
            return json_error("processing_model takes no view, provider_id or harness_id", 400)
        from ouroboros.pricing import route_processing

        return JSONResponse(await asyncio.to_thread(route_processing, str(query["processing_model"]), load_settings))
    source_id = str(query.get("source_id") or "").strip()
    profile_id = str(query.get("credential_profile_id") or "").strip()
    if query.get("view") == "sources":
        if source_id or profile_id or "provider_id" in query or "harness_id" in query:
            return json_error("view=sources cannot be combined with a source, profile or provider selector", 400)
        return JSONResponse(await asyncio.to_thread(_subscription_model_catalog, sources_only=True))
    if "provider_id" in query:
        provider_id = query["provider_id"].strip()
        if not provider_id or source_id or profile_id or "harness_id" in query:
            return json_error("provider_id must name one connection without a subscription selector", 400)
        return JSONResponse(await catalog_payload(load_settings(), provider_id=provider_id))
    if "harness_id" in query:
        from ouroboros.gateway.claudexor_accounts import session_model_catalog

        harness_id = query["harness_id"].strip()
        if not harness_id or source_id:
            return json_error("harness_id must name one agent session without a subscription source", 400)
        return JSONResponse(await asyncio.to_thread(session_model_catalog, harness_id, profile_id))
    if profile_id and not source_id:
        return json_error("credential_profile_id requires source_id or harness_id", 400)
    if source_id:
        return JSONResponse(await asyncio.to_thread(_subscription_model_catalog, source_id, profile_id))
    return JSONResponse(await catalog_payload(load_settings()))


async def draft_catalog(saved: dict, draft: dict, typed: set, *, provider_id: str | None = None) -> dict:
    """The catalog of an unsaved settings draft (the setup wizard's): ``catalog_payload`` over ``draft``, recording
    no capability evidence. A connection whose kept secret the draft would send somewhere it was not saved for is
    not asked: its reason is its error row (``_saved_secret_moved``; ``typed`` is what the draft's author typed)."""
    return await catalog_payload(draft, provider_id=provider_id, refused=lambda source_id: _saved_secret_moved(
        source_id, *_kept_binding(source_id, saved, draft), typed))


def _kept_binding(source_id: str, saved: dict, draft: dict) -> tuple[dict, dict]:
    """The saved and draft settings ``_saved_secret_moved`` compares for one connection of a draft, prepared as
    Provider Test prepares them: the kept key's binding is its own saved row or pair, whatever serves today. A saved
    row counts as switched on (a draft may only switch it on), and the built-in compatible connection is read as
    the one pair the draft reaches, so the other pair's saved address is never taken for this key's."""
    from ouroboros.llm_routing import resolve_target

    route = parse_model_route(f"{source_id}::")
    own = (route.kind, route.connection or BUILTIN_CONNECTION)
    saved = {**saved, CONNECTIONS_KEY: [{**row, "enabled": True} if (row.get("kind"), row.get("name")) == own else row
                                        for row in connection_rows(saved.get(CONNECTIONS_KEY))]}
    if own != ("openai-compatible", BUILTIN_CONNECTION):
        return saved, draft

    def reached(settings: dict) -> tuple:
        try:
            target = resolve_target(f"{source_id}::probe", settings)
        except Exception:
            return ()
        return str(target.get("api_key") or ""), str(target.get("base_url") or "")

    pairs = [(_pair_test_settings(saved, {}, key), _pair_test_settings(draft, {}, key)) for key in _COMPATIBLE_PAIRS]
    return next((pair for pair in pairs if reached(pair[1]) and reached(pair[1]) == reached(draft)), (saved, draft))


async def catalog_payload(settings: dict, *, provider_id: str | None = None,
                          refused: Callable[[str], str] | None = None) -> dict:
    """Read independent inventories together. A selected API connection needs no subscription read.
    A draft names refused connections and records no capability evidence."""
    if provider_id is not None:
        return await _api_catalog(settings, provider_id, refused)
    # A whole-list client has one response budget. Slow subscription discovery must not consume
    # the API connections' time before they even start, or discard their already returned models.
    async def subscriptions():
        batch = _SubscriptionBatch()
        try:
            return await asyncio.wait_for(asyncio.to_thread(_subscription_model_catalog, "", "", batch), _CATALOG_HTTP_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            return batch.snapshot()
    engine, api = await asyncio.gather(subscriptions(), _api_catalog(settings, None, refused))
    items = sorted(engine["items"] + api["items"], key=lambda item: (item.get("provider", "").lower(), item.get("name", "").lower()))
    return {**engine, "items": items, "errors": engine["errors"] + api["errors"], "api_sources": api["api_sources"]}


async def _api_catalog(settings, provider_id, refused):
    result = {"items": [], "errors": [], "model_sources": [], "api_sources": []}
    if provider_id == "local":
        return {**result, "items": list(filter(None, [_local_model_entry(settings)]))}
    items = result["items"]
    errors = result["errors"]
    kept_at: dict[str, int] = {}
    configured = _provider_specs(settings, record_image_input=refused is None, provider_id=provider_id)
    if provider_id is not None and not configured:
        return {**result, "errors": [{"provider_id": provider_id, "stage": "configuration",
                                      "error": "This connection has no configured model catalog."}]}
    specs = []
    for source, loader in configured:
        if refused and (reason := refused(source["id"])):
            errors.append({"provider_id": source["id"], "error": reason, "stage": "kept_key", "duration_ms": 0})
        else:
            specs.append((source, loader))

    timeout = httpx.Timeout(_CATALOG_HTTP_TIMEOUT_SEC)
    try:
        verify = verify_kwargs()
    except ExtraCaBundleError as exc:
        # A misconfigured trust bundle is a fact for the owner to read, not a bare 500:
        # keep the engine catalog, name the setting, skip the API providers this turn.
        errors.append({"provider_id": provider_id or "extra_ca_bundle", "error": str(exc), "stage": "trust", "duration_ms": 0})
        results = []
    else:
        async with httpx.AsyncClient(timeout=timeout, **verify) as client:
            results = await asyncio.gather(*[
                _load_provider(client, source["id"], loader)
                for source, loader in specs
            ])

    for provider_items, error in results:
        if error:
            errors.append(error)
            continue
        for item in provider_items:
            value = str(item.get("value", "") or "")
            if value in kept_at:
                kept = items[kept_at[value]]
                # Records that disagree on a price or its condition show none of either.
                if any(key in kept and kept[key] != item.get(key) for key in ("price", "price_condition")):
                    items[kept_at[value]] = {key: field for key, field in kept.items()
                                             if key not in ("price", "price_condition")}
                continue
            if value:
                kept_at[value] = len(items)
                items.append(item)

    if provider_id is None:
        items.extend(filter(None, [_local_model_entry(settings)]))
    items.sort(key=lambda item: (item.get("provider", "").lower(), item.get("name", "").lower()))
    return {**result, "items": items, "errors": errors, "api_sources": [source for source, _ in specs]}


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
        from ouroboros.deployment_policy import local_launch_refusal
        from ouroboros.gateway.owner_settings import MANAGED_BY_DEPLOYMENT

        # The deployment judges the launch before any probe, download or start; Stop stays free.
        if refusal := local_launch_refusal(source=source, filename=filename, port=port, chat_format=chat_format):
            return json_error(refusal, 403, code=MANAGED_BY_DEPLOYMENT)

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

        mgr.start_server(model_path, port=port, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                         chat_format=chat_format, source=source, filename=filename)
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
        return JSONResponse({**mgr.status_dict(), "settings_application": mgr.settings_application(load_settings())})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


async def api_local_model_test(request: Request) -> JSONResponse:
    try:
        from ouroboros.local_model import get_manager
        mgr = get_manager()
        if not mgr.is_running:
            return json_error("Local model server is not running", 400)
        # Queueing and the owned probe must not hold the control loop or leave with the HTTP waiter.
        result = await run_sync_to_completion(mgr.test_tool_calling)
        return JSONResponse(result)
    except Exception as e:
        return json_exception(e)


_DEDICATED_KEY, _DEDICATED_URL = (PROVIDER_KINDS["openai-compatible"].flat_key(field) for field in ("api_key", "base_url"))
# The built-in compatible connection's two pairs, key setting → address setting (``compatible_endpoint``).
# Discovery and Test name the pair a draft reaches, since both may be saved, even for one address: a kept
# key is read from the named pair and never guessed from an address.
_COMPATIBLE_PAIRS = {_DEDICATED_KEY: _DEDICATED_URL,
                     COMPATIBLE_LEGACY_PAIR[_DEDICATED_KEY]: COMPATIBLE_LEGACY_PAIR[_DEDICATED_URL]}


async def api_openai_compatible_models(request: Request) -> JSONResponse:
    """Proxy the connection's model list so the onboarding wizard avoids browser CORS limits."""
    try:
        body = await request.json()
        base_url = str(body.get("baseUrl", "") or "").strip()
        api_key = str(body.get("apiKey", "") or "").strip()
        if not base_url:
            return json_error("baseUrl is required", 400)
        name = str(body.get("connection") or "").strip()
        key_setting = str(body.get("apiKeySetting") or "").strip()
        if key_setting and (name or key_setting not in _COMPATIBLE_PAIRS):
            return json_error(f"apiKeySetting must be {' or '.join(_COMPATIBLE_PAIRS)}, for the built-in pair only", 400)
        # A wizard reopening an install holds a masked key, just as Settings does.
        # Resolve only the selected identity, for the address it was saved with: any page can post
        # here, so a placeholder never carries a key elsewhere. An explicit blank still means no auth.
        from ouroboros.secret_masking import looks_masked_settings_secret
        if looks_masked_settings_secret(key_setting or _DEDICATED_KEY, api_key):
            from ouroboros.gateway.secret_rows import rehydrate_connections
            settings = dict(load_settings())
            if name:
                draft = {"kind": "openai-compatible", "name": name, "api_key": api_key}
                api_key = rehydrate_connections([draft], settings.get(CONNECTIONS_KEY))[0]["api_key"]
                saved_url = (find_row(connection_rows(settings.get(CONNECTIONS_KEY)), "openai-compatible", name) or {}).get("base_url")
            elif key_setting:
                api_key = str(settings.get(key_setting) or "").strip()
                saved_url = settings.get(_COMPATIBLE_PAIRS[key_setting])
            else:
                from ouroboros.llm_routing import resolve_target
                # Explicit discovery also works while the built-in source is off.
                settings[CONNECTIONS_KEY] = [row for row in connection_rows(settings.get(CONNECTIONS_KEY))
                                             if (row.get("kind"), row.get("name")) != ("openai-compatible", BUILTIN_CONNECTION)]
                target = resolve_target("openai-compatible::catalog", settings)
                api_key, saved_url = target["api_key"], target.get("base_url")
            # Provider Test's endpoint identity (``_saved_secret_moved``): one key-to-address binding for both.
            if api_key and _endpoint_identity(saved_url) != _endpoint_identity(base_url):
                return json_error("This server's address changed since its key was saved. Enter its key again "
                                  "to list its models.", 400, code="masked_key_endpoint_changed")
        async with httpx.AsyncClient(timeout=10.0, **verify_kwargs()) as client:
            models = await _fetch_openai_compatible_model_catalog(
                client, "openai-compatible", "OpenAI-compatible", api_key, base_url
            )
        model_ids = [parse_model_route(m["value"]).model for m in models if m.get("value")]
        return JSONResponse({"models": model_ids})
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": f"HTTP {e.response.status_code}"}, status_code=502)
    except Exception as e:
        return json_exception(e)


_PROVIDER_TEST_KNOWN_IDS: frozenset[str] = frozenset({
    "openrouter", "openai-compatible", *DIRECT_PROVIDER_DEFAULTS,
})


def _provider_test_model(provider_id: str, settings: dict, connection: str = "") -> str:
    """One deterministic runtime/default model of the kind, without catalog I/O, spelled
    on the connection under test (``connection`` '' = the built-in one). A route on that
    connection wins; a fixed provider serves the same ids on every key, while another
    compatible connection is another server with its own models (catalog discovery). The web search
    Model is no chat route: in its own grammar a bare id is OpenAI-native (``search_routes``)."""
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS

    routes = [route for key in ACTIVE_MODEL_SETTING_KEYS if key != MODEL_ROLE_SETTINGS["websearch"]
              for route in map(parse_model_route, str(settings.get(key, "") or "").split(","))
              if route.kind == provider_id and route.valid and route.model.strip()]
    own = [route for route in routes if route.connection == connection]
    # Another key's route only where it proposes the same profile as the key under test: an OpenAI key on
    # Chat Completions never borrows a model its peer runs on Responses (``direct_provider_defaults``).
    profile = direct_provider_defaults(provider_id, settings, connection)
    shared = [] if provider_id == "openai-compatible" else [
        route for route in routes if direct_provider_defaults(provider_id, settings, route.connection) == profile]
    native = next((route.model.strip() for route in own or shared), "")
    if not native:
        default = OPENROUTER_DEFAULTS if provider_id == "openrouter" else profile  # its connection's own profile
        native = parse_model_route(str(default.get("main") or "").strip()).model.strip()
    if not native or (provider_id == "openrouter" and not connection):
        return native
    return f"{connection_id(provider_id, connection)}::{native}"


def _discover_provider_test_model(source_id: str, settings: dict) -> str:
    """Catalog discovery for the sole kind without a maintained default (a compatible connection)."""
    if parse_model_route(f"{source_id}::").kind != "openai-compatible":
        return ""

    async def load_first() -> str:
        loader = next((loader for source, loader in _provider_specs(settings) if source["id"] == source_id), None)
        if loader is None:
            return ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(_CATALOG_HTTP_TIMEOUT_SEC), **verify_kwargs()) as client:
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
        **{key: result[key] for key in ("requested", "applied") if key in result},  # the tool test's forms
    }).value
    logger = log.info if facts.get("ok") else log.warning
    logger("provider_test result=%s", facts)


def _run_provider_test_with_settings(provider_id: str, settings: dict, connection: str = "") -> dict:
    """Run selection, optional discovery, client construction and one send."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_probe import controlled_probe_error

    started = time.perf_counter()
    model = ""
    try:
        model = _provider_test_model(provider_id, settings, connection)
        if not model:
            model = _discover_provider_test_model(connection_id(provider_id, connection), settings)
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


def _run_provider_test(provider_id: str, overrides: dict[str, str], connection: str = "", key_setting: str = "") -> dict:
    """Load and merge the request-local draft inside the worker thread.

    Test is the owner's explicit request, so a switched-off connection is tested too.
    A named connection's draft fields override its saved row (a row not saved yet is
    tested from the draft alone, on the OpenAI API its save would seed:
    ``provider_connections.seed_new_api_surfaces``); an unchanged secret's placeholder keeps the saved one,
    on the built-in connection too. ``key_setting`` names the built-in compatible pair
    under test (``_pair_test_settings``). A kept secret reaches only its saved endpoint
    (``_saved_secret_moved``): before discovery and the probe, so nothing is sent."""
    from ouroboros.secret_masking import looks_masked_settings_secret

    settings = dict(load_settings())
    before = dict(settings)  # the read a save of this draft would start from
    saved_rows = connection_rows(settings.get(CONNECTIONS_KEY))
    under_test = (provider_id, connection or BUILTIN_CONNECTION)
    rows = [row for row in saved_rows if (row.get("kind"), row.get("name")) != under_test]
    secrets = {field if connection else flat: flat for field, flat, secret in PROVIDER_KINDS[provider_id].fields if secret}
    if key_setting:  # the legacy pair's OPENAI_API_KEY is no field of this kind
        secrets[key_setting] = key_setting
    fresh = {value.strip() for key, value in overrides.items() if key in secrets and value.strip()
             and not looks_masked_settings_secret(secrets[key], value)}
    draft = {key: value.strip() for key, value in overrides.items()
             if not (key in secrets and looks_masked_settings_secret(secrets[key], value))}
    saved_row = find_row(saved_rows, provider_id, connection) if connection else None
    saved = {**settings, CONNECTIONS_KEY: [*rows, *([{**saved_row, "enabled": True}] if saved_row else [])]}
    if connection:
        row = {**(saved_row or {"kind": provider_id, "name": connection}), **draft, "enabled": True}
        rows.append(row)
        read = row.get
    else:
        if key_setting:  # the kept key's binding is the named pair's saved address, too
            saved, settings = _pair_test_settings(saved, {}, key_setting), _pair_test_settings(settings, draft, key_setting)
        else:
            settings = _builtin_test_settings(provider_id, settings, draft)
        read = lambda field: settings.get(PROVIDER_KINDS[provider_id].flat_key(field))  # noqa: E731
    problem = next((field for field in ("region", "plan", "api_surface") if choice_error(provider_id, field, read(field))), "")
    if problem:
        return {"error": {"region": "unknown MiniMax region", "plan": "unknown Z.ai plan",
                          "api_surface": "unknown OpenAI API"}[problem], "_http_status": 400}
    settings[CONNECTIONS_KEY] = rows
    seed_new_api_surfaces(settings, before)  # a draft not saved yet is tested on the API its save would store
    # A named pair is as ready as the runtime reads it: a legacy address without its key sends nothing.
    if key_setting and not compatible_endpoint_configured(lambda key: str(settings.get(key) or "").strip()):
        return {"ok": False, "error": "Provider is not configured"}
    if moved := _saved_secret_moved(connection_id(provider_id, connection), saved, settings, fresh):
        return {"error": moved, "code": "masked_key_endpoint_changed", "_http_status": 400}
    return _run_provider_test_with_settings(provider_id, settings, *((connection,) if connection else ()))


# Kinds whose endpoint the owner types; the others' endpoints are the vendor's own
# (a MiniMax region, Z.ai plan or OpenAI API picks among its closed set, ``choice_error``).
_ADDRESSED_KINDS = frozenset(kind for kind, spec in PROVIDER_KINDS.items() if spec.flat_key("base_url"))


def _endpoint_identity(base_url: object) -> tuple:
    """Where requests built on a base URL go. The OpenAI SDK and httpx (the GigaChat library)
    append each path after one enforced trailing slash and catalog reads strip it, so one
    trailing slash, case, a default port and dot segments change nothing. They append after
    a query too, and a catalog path lands inside a fragment, so a URL with either is the
    same endpoint only when spelled the same."""
    text = str(base_url or "").strip()
    try:
        url = httpx.URL(text)
    except Exception:
        return ("verbatim", text)
    if "?" in text or "#" in text:
        return ("verbatim", text)
    path = url.raw_path.decode("ascii", "replace")
    port = url.port or {"http": 80, "https": 443}.get(url.scheme)
    return (url.scheme, url.userinfo, url.host, port, path[:-1] if path.endswith("/") else path)


def _sent_secrets(target: dict) -> list[str]:
    """The secrets a target's transport sends: its key, except that the GigaChat library
    refreshes its token with the authorization key whenever one is set, and only otherwise
    with ``user`` and ``password`` (``GigaChat._update_token``)."""
    def text(field: str) -> str:
        return str(target.get(field) or "").strip()

    if target.get("provider") == "gigachat" and not text("api_key"):
        return [text("password")] if text("user") and text("password") else []
    return [text("api_key")] if text("api_key") else []


def _saved_secret_moved(source_id: str, saved: dict, draft: dict, fresh: set[str]) -> str:
    """Why the draft would send a saved secret somewhere it was not saved for, or ''.

    Both sides resolve through the real route (``resolve_target``). A secret typed in
    this request may go anywhere; a kept one (omitted or masked) that the transport
    would send goes only to the same scheme, host, port and path with no weaker TLS
    check. Saving the new address remains the owner's way to move a saved key."""
    from ouroboros.llm_routing import resolve_target

    if parse_model_route(f"{source_id}::").kind not in _ADDRESSED_KINDS:
        return ""
    try:
        target = resolve_target(f"{source_id}::probe", draft)
    except Exception:
        return ""  # nothing can be sent without a target
    kept = [secret for secret in _sent_secrets(target) if secret not in fresh]
    if not kept or not str(target.get("base_url") or "").strip():
        return ""
    try:
        before = resolve_target(f"{source_id}::probe", saved)
    except Exception:
        before = {}
    if _endpoint_identity(before.get("base_url")) != _endpoint_identity(target.get("base_url")):
        return "This server's address changed since its key was saved. Enter its key again to test the new address."
    if before.get("verify_ssl_certs", True) and not target.get("verify_ssl_certs", True):
        return "Certificate checking was turned off since this key was saved. Enter its key again to test without it."
    return ""


def _pair_test_settings(settings: dict, overrides: dict[str, str], key_setting: str) -> dict:
    """Exactly the built-in compatible pair ``key_setting`` names, under the request's edits, the other pair
    blanked: the probe and the kept key's binding read that pair alone, as the runtime reads the pair a
    draft reaches (``compatible_endpoint``), with no historical fallback and no key from the other pair."""
    pair = (key_setting, _COMPATIBLE_PAIRS[key_setting])
    other = [key for item in _COMPATIBLE_PAIRS.items() if item != pair for key in item]
    return {**settings, **{key: value for key, value in overrides.items() if key in pair}, **dict.fromkeys(other, "")}


def _builtin_test_settings(provider_id: str, settings: dict, overrides: dict[str, str]) -> dict:
    """The built-in connection's draft: flat-key overrides over the saved settings. A request
    without ``api_key_setting`` keeps this historical compatible resolution."""
    if provider_id == "openai-compatible":
        # No-edit requests retain the historical OPENAI_* fallback. An explicit
        # Clear on the dedicated card must not silently restore the corresponding
        # legacy value for this request-local probe.
        legacy_fallbacks = COMPATIBLE_LEGACY_PAIR
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
    return settings


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
    # A named connection is tested with its row's draft fields; the built-in one with flat keys.
    connection = str(body.get("connection") or "").strip()
    connection = "" if connection == BUILTIN_CONNECTION else connection
    if connection and not parse_model_route(f"{provider_id}@{connection}::").valid:
        return json_error("connection must be a name of 1-32 lowercase letters, digits, '-' or '_'", 400)

    # The built-in compatible pair a draft reaches, named by its key setting; absent = the historical resolution.
    key_setting = str(body.get("api_key_setting") or "")
    if key_setting and (provider_id != "openai-compatible" or connection or key_setting not in _COMPATIBLE_PAIRS):
        return json_error(f"api_key_setting must be {' or '.join(_COMPATIBLE_PAIRS)}, for the built-in "
                          "OpenAI-compatible pair only", 400)

    overrides = body.get("overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        return json_error("overrides must be an object", 400)
    # Only the tested connection's own fields (and its named pair's): another kind's key never rides this probe.
    allowed = {field if connection else flat for field, flat, _secret in PROVIDER_KINDS[provider_id].fields}
    allowed.update((key_setting, _COMPATIBLE_PAIRS[key_setting]) if key_setting else ())
    unknown = sorted(str(key) for key in overrides if str(key) not in allowed)
    if unknown:
        return json_error(f"unsupported override keys: {', '.join(unknown)}", 400)
    if any(not isinstance(value, str) for value in overrides.values()):
        return json_error("override values must be strings", 400)
    # A dedicated address selects the dedicated pair, a blank one the legacy pair: the request names that one.
    if key_setting and _DEDICATED_URL in overrides and bool(overrides[_DEDICATED_URL].strip()) != (key_setting == _DEDICATED_KEY):
        return json_error(f"api_key_setting names the pair this draft reaches: {_DEDICATED_KEY} with a "
                          f"{_DEDICATED_URL}, else {COMPATIBLE_LEGACY_PAIR[_DEDICATED_KEY]}", 400)

    result = await asyncio.to_thread(
        _run_provider_test,
        provider_id,
        {str(key): value for key, value in overrides.items()},
        *((connection,) if connection else ("", key_setting) if key_setting else ()),
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
        from ouroboros.deployment_policy import body_replacement_refusal
        from ouroboros.gateway.owner_settings import MANAGED_BY_DEPLOYMENT
        from ouroboros.local_model import get_manager

        # pip writes into the interpreter this server runs: a managed deployment's administrator provisions it.
        if refusal := body_replacement_refusal():
            return json_error(refusal, 403, code=MANAGED_BY_DEPLOYMENT)
        mgr = get_manager()

        current = mgr._runtime_status
        if current == "installing":
            return JSONResponse({"status": "already_installing"})

        mgr.install_runtime()
        return JSONResponse({"status": "installing"})
    except Exception as e:
        return json_exception(e)
