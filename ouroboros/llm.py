"""LLM client for OpenRouter, direct providers, and optional local inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.provider_models import normalize_anthropic_model_id, normalize_model_identity
from ouroboros.utils import in_worker_process

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3.5-flash"
_FALSE_LIKE_ENV_VALUES = {"", "0", "false", "no", "off"}
_OPTIONAL_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


class LocalContextTooLargeError(RuntimeError):
    """Raised when a local model cannot fit context without silent truncation."""


def _estimate_message_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            total += sum(len(str(block.get("text", ""))) for block in content if isinstance(block, dict))
        else:
            total += len(str(content or ""))
    return total


def _split_markdown_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    lines = str(text or "").splitlines()
    preamble: List[str] = []
    sections: List[Tuple[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_title is None:
                preamble = current_lines[:]
            else:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_title is None:
        return "\n".join(lines).strip(), []

    sections.append((current_title, "\n".join(current_lines).strip()))
    return "\n".join(preamble).strip(), sections


def _compact_markdown_sections(
    text: str,
    preserve_titles: Set[str],
    reason: str,
) -> str:
    preamble, sections = _split_markdown_sections(text)
    if not sections:
        return text

    parts: List[str] = []
    if preamble:
        parts.append(preamble)

    for title, section in sections:
        if title in preserve_titles:
            parts.append(section)
            continue
        omitted_chars = max(0, len(section))
        parts.append(
            f"## {title}\n\n"
            f"[Compacted for local-model context: omitted {omitted_chars} chars. {reason}]"
        )

    return "\n\n".join(p for p in parts if p).strip()


_LOCAL_COMPACTION_MODES = {
    "static": (
        {"BIBLE.md"},
        "Use a larger-context model or read the source file directly if this section becomes necessary.",
    ),
    "semi_stable": (
        {"Identity"},
        "Identity was preserved; non-core stable memory sections were compacted for local execution.",
    ),
    "dynamic": (
        {
            "Scratchpad",
            "Dialogue History",
            "Dialogue Summary",
            "Memory Registry (what I know / don't know)",
            "Drive state",
            "Runtime context",
            "Health Invariants",
        },
        "Working-memory and runtime sections were preserved; non-core recent/history sections were compacted for local execution.",
    ),
    "system": (
        {
            "BIBLE.md",
            "Scratchpad",
            "Identity",
            "Drive state",
            "Runtime context",
            "Health Invariants",
            "Recent observations",
            "Background consciousness info",
        },
        "Non-core sections were compacted for local execution.",
    ),
}


def _compact_local_text(text: str, mode: str) -> str:
    preserve_titles, reason = _LOCAL_COMPACTION_MODES[mode]
    return _compact_markdown_sections(text, preserve_titles=preserve_titles, reason=reason)


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, ...]]:
    """Fetch OpenRouter pricing as model_id -> per-1M prices.

    Tuples are ``(input, cached_read, output)`` or
    ``(input, cached_read, cache_write, output)`` when OpenRouter exposes a
    provider-specific write price.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None
            raw_cache_write_str = pricing.get("input_cache_write")
            raw_cache_write = float(raw_cache_write_str) if raw_cache_write_str else None

            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                # Missing cache-read pricing is not a provider promise. Use the
                # conservative input price unless the response carries an
                # authoritative usage.cost value.
                cached_price = prompt_price
            cache_write_price = (
                round(raw_cache_write * 1_000_000, 4)
                if raw_cache_write is not None else None
            )

            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            if cache_write_price is not None:
                row = (prompt_price, cached_price, cache_write_price, completion_price)
            else:
                row = (prompt_price, cached_price, completion_price)
            pricing_dict[model_id] = row
            normalized_model_id = normalize_model_identity(model_id)
            if normalized_model_id != model_id:
                pricing_dict[normalized_model_id] = row

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Routes calls to OpenRouter or a local llama-cpp-python server."""

    # Missing capabilities mean "unknown": keep kwargs instead of stripping them.
    _SUPPORTED_PARAMS_CACHE: Dict[str, set] = {}
    _SUPPORTED_PARAMS_FETCHED: bool = False
    _REJECTED_PARAMS_CACHE: Dict[str, Set[str]] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key_override = api_key
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None
        self._client_api_key: Optional[str] = None
        self._async_client = None
        self._async_client_api_key: Optional[str] = None
        self._local_client = None
        self._local_port: Optional[int] = None
        self._remote_clients: Dict[Tuple[str, str, str, Tuple[Tuple[str, str], ...]], Any] = {}
        self._async_remote_clients: Dict[Tuple[str, str, str, Tuple[Tuple[str, str], ...]], Any] = {}
        self._gigachat_clients: Dict[Tuple[str, str, str, str, str, bool], Any] = {}

    @classmethod
    def _fetch_openrouter_capabilities(cls) -> None:
        """Populate _SUPPORTED_PARAMS_CACHE once from OpenRouter /models."""
        cls._SUPPORTED_PARAMS_FETCHED = True
        try:
            import requests
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=15,
            )
            if resp.status_code != 200:
                log.debug(
                    "OpenRouter /models returned %d; supported_parameters cache empty",
                    resp.status_code,
                )
                return
            for m in resp.json().get("data", []) or []:
                mid = m.get("id") or ""
                sp = m.get("supported_parameters")
                if mid and isinstance(sp, list) and sp:
                    cls._SUPPORTED_PARAMS_CACHE[mid] = set(sp)
        except Exception:
            log.debug("Failed to fetch OpenRouter model capabilities", exc_info=True)

    @classmethod
    def _get_supported_parameters(cls, model_id: str) -> Optional[set]:
        """Return supported parameter names, or None when unknown/no stripping."""
        if not cls._SUPPORTED_PARAMS_FETCHED:
            cls._fetch_openrouter_capabilities()
        return cls._SUPPORTED_PARAMS_CACHE.get(model_id)

    @staticmethod
    def _parameter_rejection_error(exc: BaseException) -> bool:
        text = str(exc or "").lower()
        if not text:
            return False
        # OpenRouter rejects unsupported sampling params (with require_parameters)
        # as "No endpoints found that support the requested parameters: ...".
        # Require an explicit parameter signal so unrelated "no endpoints found"
        # errors (e.g. "...that support tool use") do not falsely match.
        if "no endpoints found" in text and (
            "requested parameter" in text
            or any(param in text for param in _OPTIONAL_SAMPLING_PARAMS)
        ):
            return True
        if not any(param in text for param in _OPTIONAL_SAMPLING_PARAMS):
            return False
        return any(
            marker in text
            for marker in (
                "unsupported",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "deprecated",
                "invalid parameter",
                "extraneous",
            )
        )

    @classmethod
    def _remember_rejected_params(cls, model_id: str, params: Set[str]) -> None:
        if not model_id or not params:
            return
        keys = {model_id, normalize_model_identity(model_id)}
        for key in keys:
            if not key:
                continue
            existing = cls._REJECTED_PARAMS_CACHE.setdefault(key, set())
            existing.update(params)

    @classmethod
    def _known_rejected_params(cls, model_id: str) -> Set[str]:
        if not model_id:
            return set()
        out: Set[str] = set()
        for key in {model_id, normalize_model_identity(model_id)}:
            out.update(cls._REJECTED_PARAMS_CACHE.get(key, set()))
        return out

    @classmethod
    def _apply_rejected_param_cache(cls, payload: Dict[str, Any], model_id: str) -> None:
        for param in cls._known_rejected_params(model_id):
            payload.pop(param, None)

    @classmethod
    def _retry_without_optional_sampling(
        cls,
        payload: Dict[str, Any],
        model_id: str,
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        if not cls._parameter_rejection_error(exc):
            return None
        present = {param for param in _OPTIONAL_SAMPLING_PARAMS if param in payload}
        if not present:
            return None
        cls._remember_rejected_params(model_id, present)
        retry_payload = copy.deepcopy(payload)
        for param in present:
            retry_payload.pop(param, None)
        log.warning(
            "Retrying %s without optional sampling parameter(s): %s",
            model_id or "(unknown model)",
            ", ".join(sorted(present)),
        )
        return retry_payload

    @staticmethod
    def _parse_provider_model(model: str) -> Tuple[str, str]:
        model_name = str(model or "").strip()
        for prefix, provider in (
            ("openai::", "openai"),
            ("anthropic::", "anthropic"),
            ("cloudru::", "cloudru"),
            ("gigachat::", "gigachat"),
            ("openai-compatible::", "openai-compatible"),
            ("openrouter::", "openrouter"),
        ):
            if model_name.startswith(prefix):
                return provider, model_name[len(prefix):].strip()
        return "openrouter", model_name

    @staticmethod
    def _qualified_model_name(provider: str, resolved_model: str) -> str:
        if provider == "openrouter":
            return resolved_model
        if provider == "openai":
            return f"openai/{resolved_model}"
        if provider == "anthropic":
            return f"anthropic/{resolved_model}"
        if provider == "cloudru":
            return f"cloudru/{resolved_model}"
        if provider == "gigachat":
            return f"gigachat/{resolved_model}"
        return f"openai-compatible/{resolved_model}"

    def _resolve_remote_target(self, model: str) -> Dict[str, Any]:
        provider, resolved_model = self._parse_provider_model(model)
        usage_model = self._qualified_model_name(provider, resolved_model)

        if provider == "openai":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1",
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "anthropic":
            resolved_model = normalize_anthropic_model_id(resolved_model)
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": self._qualified_model_name(provider, resolved_model),
                "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "base_url": "https://api.anthropic.com/v1",
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "cloudru":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", ""),
                "base_url": (
                    os.environ.get("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "") or ""
                ).strip() or "https://foundation-models.api.cloud.ru/v1",
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "gigachat":
            # GigaChat is NOT OpenAI-compatible — the `gigachat` library owns
            # the transport and auth. Everything is env-configurable: `api_key`
            # holds the authorization key (base64 client_id:secret) for the OAuth
            # flow, OR user/password for basic auth against an internal endpoint.
            # base_url/scope/verify are carried for the `_chat_gigachat` path.
            verify_raw = (os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "") or "").strip().lower()
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": os.environ.get("GIGACHAT_CREDENTIALS", ""),
                "user": (os.environ.get("GIGACHAT_USER", "") or "").strip(),
                "password": os.environ.get("GIGACHAT_PASSWORD", "") or "",
                "base_url": (
                    os.environ.get("GIGACHAT_BASE_URL", "") or ""
                ).strip() or "https://gigachat.devices.sberbank.ru/api/v1",
                "scope": (os.environ.get("GIGACHAT_SCOPE", "") or "").strip() or "GIGACHAT_API_PERS",
                "verify_ssl_certs": verify_raw not in ("0", "false", "no", "off"),
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "openai-compatible":
            compatible_key = (os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
            compatible_base_url = (os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip()
            legacy_base_url = (os.environ.get("OPENAI_BASE_URL", "") or "").strip()
            legacy_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": compatible_key or legacy_key,
                "base_url": compatible_base_url or legacy_base_url,
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        current_api_key = self._api_key_override
        if current_api_key is None:
            current_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return {
            "provider": "openrouter",
            "resolved_model": resolved_model,
            "usage_model": usage_model,
            "api_key": current_api_key,
            "base_url": self._base_url,
            "default_headers": {
                "HTTP-Referer": "https://ouroboros.local/",
                "X-Title": "Ouroboros",
            },
            "supports_openrouter_extensions": True,
            "supports_generation_cost": True,
        }

    def _get_client(self):
        target = self._resolve_remote_target("openrouter::")
        return self._get_remote_client(target)

    def _get_remote_client(self, target: Dict[str, Any]):
        base_url = str(target.get("base_url") or "")
        api_key = str(target.get("api_key") or "")
        headers_dict = dict(target.get("default_headers") or {})
        headers = tuple(sorted((str(k), str(v)) for k, v in headers_dict.items()))
        cache_key = (str(target.get("provider") or ""), base_url, api_key, headers)

        client = self._remote_clients.get(cache_key)
        if client is None:
            from openai import OpenAI

            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if headers_dict:
                kwargs["default_headers"] = headers_dict
            client = OpenAI(**kwargs)
            self._remote_clients[cache_key] = client
        return client

    def _get_local_client(self):
        port = int(os.environ.get("LOCAL_MODEL_PORT", "8766"))
        if self._local_client is None or self._local_port != port:
            from openai import OpenAI
            self._local_client = OpenAI(
                base_url=f"http://127.0.0.1:{port}/v1",
                api_key="local",
                max_retries=0,
            )
            self._local_port = port
        return self._local_client

    def _get_async_remote_client(self, target: Dict[str, Any]):
        base_url = str(target.get("base_url") or "")
        api_key = str(target.get("api_key") or "")
        headers_dict = dict(target.get("default_headers") or {})
        headers = tuple(sorted((str(k), str(v)) for k, v in headers_dict.items()))
        cache_key = (str(target.get("provider") or ""), base_url, api_key, headers)

        client = self._async_remote_clients.get(cache_key)
        if client is None:
            from openai import AsyncOpenAI

            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            if headers_dict:
                kwargs["default_headers"] = headers_dict
            client = AsyncOpenAI(**kwargs)
            self._async_remote_clients[cache_key] = client
        return client

    @staticmethod
    def _no_proxy_timeout():
        import httpx

        return httpx.Timeout(connect=30.0, read=3600.0, write=3600.0, pool=30.0)

    @classmethod
    def _make_no_proxy_client(cls, target: Dict[str, Any]):
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(
            trust_env=False,
            mounts={},
            timeout=cls._no_proxy_timeout(),
        )
        oa_client = OpenAI(
            api_key=str(target.get("api_key") or ""),
            base_url=str(target.get("base_url") or ""),
            default_headers=dict(target.get("default_headers") or {}),
            http_client=http_client,
            max_retries=0,
        )
        return oa_client, http_client

    @classmethod
    def _make_no_proxy_async_client(cls, target: Dict[str, Any]):
        import httpx
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(
            trust_env=False,
            mounts={},
            timeout=cls._no_proxy_timeout(),
        )
        oa_client = AsyncOpenAI(
            api_key=str(target.get("api_key") or ""),
            base_url=str(target.get("base_url") or ""),
            default_headers=dict(target.get("default_headers") or {}),
            http_client=http_client,
            max_retries=0,
        )
        return oa_client, http_client

    @classmethod
    def _copy_messages_with_cache_policy(
        cls,
        messages: List[Dict[str, Any]],
        *,
        allow_message_cache_control: bool,
        flatten_tool_content_blocks: bool,
    ) -> List[Dict[str, Any]]:
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if msg.get("role") == "tool" and flatten_tool_content_blocks:
                msg["content"] = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                for block in content:
                    if isinstance(block, dict):
                        if allow_message_cache_control and isinstance(block.get("cache_control"), dict):
                            block["cache_control"] = {"type": "ephemeral"}
                        else:
                            block.pop("cache_control", None)
        return cleaned

    @staticmethod
    def _strip_openrouter_roundtrip_metadata(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip OpenRouter reasoning round-trip fields for providers that reject extra message keys."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            if msg.get("role") != "assistant":
                continue
            msg.pop("reasoning", None)
            msg.pop("reasoning_details", None)
            msg.pop("response_id", None)
        return cleaned

    @classmethod
    def _prompt_cache_ttl_from_payload(cls, *payload_parts: Any) -> Optional[str]:
        for part in payload_parts:
            items = part if isinstance(part, list) else [part]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("cache_control"), dict):
                    return "default"
                content = item.get("content")
                if isinstance(content, list) and any(
                    isinstance(block, dict) and isinstance(block.get("cache_control"), dict)
                    for block in content
                ):
                    return "default"
        return None

    def _fetch_generation_cost(
        self,
        generation_id: str,
        target: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API when usage lacks it."""
        active_target = target or self._resolve_remote_target("openrouter::")
        if not active_target.get("supports_generation_cost"):
            return None
        try:
            import requests
            base_url = str(active_target.get("base_url") or "").rstrip("/")
            api_key = str(active_target.get("api_key") or "")
            url = f"{base_url}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation cost can lag the chat response; retry once.
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 65536,
        tool_choice: str = "auto",
        use_local: bool = False,
        temperature: Optional[float] = None,
        no_proxy: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call returning (message, usage); no_proxy avoids macOS fork proxy crashes."""
        if use_local:
            return self._chat_local(messages, tools, max_tokens, tool_choice)

        # Central worker policy: any LLM call from a worker process is fork-safe
        # by default (no system proxy lookup). This covers the main agent loop,
        # consolidator, post-task threads, and supervisor dedup without each
        # call site having to remember no_proxy=True.
        no_proxy = no_proxy or in_worker_process()
        target = self._resolve_remote_target(model)
        return self._chat_remote(
            target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
            no_proxy=no_proxy,
        )

    async def chat_async(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 65536,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        no_proxy: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async remote chat; no_proxy keeps forked macOS workers off OS proxy APIs."""
        no_proxy = no_proxy or in_worker_process()
        if tools:
            raise ValueError("chat_async does not support tool calls")
        target = self._resolve_remote_target(model)
        if target.get("provider") == "anthropic":
            return await asyncio.to_thread(
                self._chat_anthropic,
                target,
                messages,
                tools,
                reasoning_effort,
                max_tokens,
                tool_choice,
                temperature,
                no_proxy,
            )
        if target.get("provider") == "gigachat":
            # The gigachat library client is synchronous; offload to a thread
            # like the Anthropic path so the event loop is never blocked.
            return await asyncio.to_thread(
                self._chat_gigachat,
                target,
                messages,
                tools,
                reasoning_effort,
                max_tokens,
                tool_choice,
                temperature,
                no_proxy,
            )
        if no_proxy:
            _oa_client, _http_client = self._make_no_proxy_async_client(target)
            try:
                kwargs = self._build_remote_kwargs(
                    target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
                    skip_capability_fetch=True,
                )
                prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
                    kwargs.get("messages"),
                    kwargs.get("tools"),
                )
                try:
                    resp = await _oa_client.chat.completions.create(**kwargs)
                except Exception as exc:
                    retry_kwargs = self._retry_without_optional_sampling(
                        kwargs,
                        str(target.get("usage_model") or target.get("resolved_model") or ""),
                        exc,
                    )
                    if retry_kwargs is None:
                        raise
                    resp = await _oa_client.chat.completions.create(**retry_kwargs)
                return self._normalize_remote_response(
                    resp.model_dump(),
                    target,
                    skip_cost_fetch=True,
                    prompt_cache_ttl=prompt_cache_ttl,
                )
            finally:
                try:
                    await _http_client.aclose()
                except Exception:
                    pass
        client = self._get_async_remote_client(target)
        kwargs = self._build_remote_kwargs(
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools
        )
        prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
            kwargs.get("messages"),
            kwargs.get("tools"),
        )
        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            retry_kwargs = self._retry_without_optional_sampling(
                kwargs,
                str(target.get("usage_model") or target.get("resolved_model") or ""),
                exc,
            )
            if retry_kwargs is None:
                raise
            resp = await client.chat.completions.create(**retry_kwargs)
        return self._normalize_remote_response(
            resp.model_dump(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )

    def _prepare_messages_for_local_context(
        self,
        messages: List[Dict[str, Any]],
        ctx_len: int,
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        available_tokens = max(256, ctx_len - max_tokens - 64)
        target_chars = available_tokens * 3
        total_chars = _estimate_message_chars(messages)
        if total_chars <= target_chars:
            return messages

        compacted = copy.deepcopy(messages)
        for msg in compacted:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for idx, block in enumerate(content):
                    if not isinstance(block, dict) or block.get("type") != "text":
                        continue
                    block_text = str(block.get("text", ""))
                    if idx == 0:
                        block["text"] = _compact_local_text(block_text, "static")
                    elif idx == 1:
                        block["text"] = _compact_local_text(block_text, "semi_stable")
                    else:
                        block["text"] = _compact_local_text(block_text, "dynamic")
            elif isinstance(content, str):
                msg["content"] = _compact_local_text(content, "system")
            break

        compacted_chars = _estimate_message_chars(compacted)
        if compacted_chars <= target_chars:
            return compacted

        raise LocalContextTooLargeError(
            f"Local model context too large after safe compaction "
            f"({compacted_chars} chars > target {target_chars})."
        )

    def _chat_local(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send a chat request to the local llama-cpp-python server."""
        client = self._get_local_client()

        clean_messages = self._strip_openrouter_roundtrip_metadata(
            self._copy_messages_with_cache_policy(
                messages,
                allow_message_cache_control=False,
                flatten_tool_content_blocks=True,
            )
        )
        local_max = min(max_tokens, 2048)
        ctx_len = 0
        try:
            from ouroboros.local_model import get_manager
            ctx_len = get_manager().get_context_length()
            if ctx_len > 0:
                local_max = min(max_tokens, max(256, ctx_len // 4))
        except Exception:
            pass

        if ctx_len > 0:
            clean_messages = self._prepare_messages_for_local_context(clean_messages, ctx_len, local_max)

        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = "\n\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        clean_tools = None
        if tools:
            clean_tools = [
                {k: v for k, v in t.items() if k != "cache_control"}
                for t in tools
            ]

        kwargs: Dict[str, Any] = {
            "model": "local-model",
            "messages": clean_messages,
            "max_tokens": local_max,
        }
        if clean_tools:
            kwargs["tools"] = clean_tools
            kwargs["tool_choice"] = tool_choice

        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                if "context_length_exceeded" in err:
                    raise LocalContextTooLargeError(err) from exc
                if attempt == 2:
                    log.warning("Local model request failed: %s", exc)
                    raise
                log.warning(
                    "Local model request failed (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(0.5 * (attempt + 1))
        if last_exc is not None:
            raise last_exc

        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        if not msg.get("tool_calls") and msg.get("content") and clean_tools:
            allowed_tool_names = {
                str(t.get("function", {}).get("name", "")).strip()
                for t in clean_tools
                if isinstance(t, dict)
            }
            msg = self._parse_tool_calls_from_content(msg, allowed_tool_names)

        usage["cost"] = 0.0
        return msg, usage

    @staticmethod
    def _strip_reasoning_wrappers(text: str):
        """Strip leading think/reasoning wrappers before the first <tool_call> only."""
        # Split at first <tool_call> so we never touch JSON inside tool payloads.
        tool_call_start = re.search(r"<tool_call\b", text, re.IGNORECASE)
        if tool_call_start:
            prefix = text[: tool_call_start.start()]
            suffix = text[tool_call_start.start():]
        else:
            prefix = text
            suffix = ""

        reasoning_parts: list = []

        def _extract(tag: str, s: str) -> str:
            pattern = re.compile(
                r"<" + re.escape(tag) + r">(.*?)</" + re.escape(tag) + r">",
                re.DOTALL | re.IGNORECASE,
            )
            inner_texts = pattern.findall(s)
            reasoning_parts.extend(p.strip() for p in inner_texts if p.strip())
            return pattern.sub("", s)

        cleaned_prefix = _extract("think", prefix)
        cleaned_prefix = _extract("reasoning", cleaned_prefix)

        combined = (cleaned_prefix.strip() + ("\n" if cleaned_prefix.strip() and suffix else "") + suffix).strip()
        return combined, "\n\n".join(reasoning_parts)

    @staticmethod
    def _parse_tool_calls_from_content(
        msg: Dict[str, Any],
        allowed_tool_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Parse local <tool_call> XML output after a strict full-match guard."""
        content = str(msg.get("content", "") or "")
        stripped_raw = content.strip()
        if not stripped_raw:
            return msg

        # Only explicit reasoning wrappers are removed; arbitrary prose is left.
        stripped, reasoning = LLMClient._strip_reasoning_wrappers(stripped_raw)
        if not stripped:
            return msg

        # Upgrade only pure tool-call output; mixed prose stays plain text.
        full_pattern = re.compile(
            r"^(?:\s*<tool_call>\s*\{.*?\}\s*</tool_call>\s*)+$",
            re.DOTALL,
        )
        if not full_pattern.fullmatch(stripped):
            return msg

        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", stripped, re.DOTALL)
        if not matches:
            return msg

        allowed = {name for name in (allowed_tool_names or set()) if name}
        tool_calls = []
        for i, raw in enumerate(matches):
            try:
                raw_stripped = raw.strip()
                try:
                    obj = json.loads(raw_stripped)
                except json.JSONDecodeError:
                    if raw_stripped.startswith("{{") and raw_stripped.endswith("}}"):
                        obj = json.loads(raw_stripped[1:-1])
                    else:
                        raise
                if not isinstance(obj, dict):
                    raise ValueError("tool_call payload must be an object")
                name = str(obj.get("name", "")).strip()
                args = obj.get("arguments", {})
                if not name:
                    raise ValueError("tool_call missing function name")
                if allowed and name not in allowed:
                    raise ValueError(f"unknown tool '{name}'")
                if not isinstance(args, dict):
                    raise ValueError("tool_call arguments must be an object")
                tool_calls.append({
                    "id": f"call_local_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            except (json.JSONDecodeError, ValueError) as exc:
                log.warning("Rejected local <tool_call> block: %s (%s)", raw[:200], exc)
                return msg

        if not tool_calls:
            return msg

        msg = dict(msg)
        msg["tool_calls"] = tool_calls
        # Preserve reasoning text for loop progress; None/empty remains falsy.
        msg["content"] = reasoning or None
        log.info("Parsed %d local tool call(s) from text output", len(tool_calls))
        return msg

    @staticmethod
    def _stringify_anthropic_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _stringify_tool_description(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "".join(str(part) for part in value if part is not None)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _coalesce_anthropic_message(
        messages: List[Dict[str, Any]],
        role: str,
        content: List[Dict[str, Any]],
    ) -> None:
        if not content:
            return
        if messages and messages[-1].get("role") == role and isinstance(messages[-1].get("content"), list):
            messages[-1]["content"].extend(content)
            return
        messages.append({"role": role, "content": list(content)})

    @staticmethod
    def _anthropic_image_block(image_url: str) -> Optional[Dict[str, Any]]:
        url = str(image_url or "").strip()
        if not url:
            return None
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(",", 1)
            mime = header[5:].split(";", 1)[0] or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    def _anthropic_blocks_from_content(self, content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if not isinstance(content, list):
            text = self._stringify_anthropic_content(content)
            return [{"type": "text", "text": text}] if text else []

        blocks: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                if block:
                    blocks.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                text = self._stringify_anthropic_content(block)
                if text:
                    blocks.append({"type": "text", "text": text})
                continue

            block_type = str(block.get("type") or "").strip()
            if block_type in {"text", "input_text", "output_text"}:
                text = str(block.get("text") or "")
                if text:
                    normalized = {"type": "text", "text": text}
                    if isinstance(block.get("cache_control"), dict):
                        normalized["cache_control"] = {"type": "ephemeral"}
                    blocks.append(normalized)
                continue
            if block_type == "image_url":
                image_url = str((block.get("image_url") or {}).get("url") or "")
                image_block = self._anthropic_image_block(image_url)
                if image_block:
                    blocks.append(image_block)
                continue
            if block.get("text"):
                normalized = {"type": "text", "text": str(block.get("text") or "")}
                if isinstance(block.get("cache_control"), dict):
                    normalized["cache_control"] = {"type": "ephemeral"}
                blocks.append(normalized)
        return blocks

    def _build_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        system_blocks: List[Dict[str, Any]] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("role") or "").strip().lower()
            if role == "system":
                system_blocks.extend(self._anthropic_blocks_from_content(msg.get("content")))
                continue

            if role == "user":
                self._coalesce_anthropic_message(
                    anthropic_messages,
                    "user",
                    self._anthropic_blocks_from_content(msg.get("content")),
                )
                continue

            if role == "assistant":
                assistant_blocks = self._anthropic_blocks_from_content(msg.get("content"))
                for tool_call in msg.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    raw_args = function.get("arguments")
                    parsed_args: Any = {}
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                        except Exception:
                            parsed_args = {"raw": raw_args}
                    elif raw_args is not None:
                        parsed_args = raw_args
                    if not isinstance(parsed_args, dict):
                        parsed_args = {"value": parsed_args}
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": parsed_args,
                    })
                self._coalesce_anthropic_message(anthropic_messages, "assistant", assistant_blocks)
                continue

            if role == "tool":
                tool_use_id = str(msg.get("tool_call_id") or "")
                if not tool_use_id:
                    raise ValueError("Anthropic direct tool result is missing tool_call_id.")
                raw_content = msg.get("content")
                # Anthropic accepts list tool_result content; stringify only scalars/dicts.
                if isinstance(raw_content, list):
                    tool_result_content: Any = self._copy_messages_with_cache_policy(
                        [{"role": "tool", "content": raw_content}],
                        allow_message_cache_control=True,
                        flatten_tool_content_blocks=False,
                    )[0]["content"]
                else:
                    tool_result_content = self._stringify_anthropic_content(raw_content)
                self._coalesce_anthropic_message(
                    anthropic_messages,
                    "user",
                    [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result_content,
                    }],
                )

        return system_blocks, anthropic_messages

    @staticmethod
    def _build_anthropic_tools(
        tools: Optional[List[Dict[str, Any]]],
        *,
        cache_control: bool = False,
    ) -> List[Dict[str, Any]]:
        anthropic_tools: List[Dict[str, Any]] = []
        for tool in LLMClient._sanitize_chat_completion_tools(tools):
            function = tool.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            anthropic_tools.append({
                "name": name,
                "description": LLMClient._stringify_tool_description(function.get("description")),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            })
        if cache_control and anthropic_tools:
            anthropic_tools[-1] = {**anthropic_tools[-1], "cache_control": {"type": "ephemeral"}}
        return anthropic_tools

    @staticmethod
    def _sanitize_chat_completion_tools(
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        sanitized_tools: List[Dict[str, Any]] = []
        seen_tool_names: Set[str] = set()
        provider_name_re = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            tool_copy = dict(tool)
            function = tool_copy.get("function") or {}
            if isinstance(function, dict):
                function_copy = dict(function)
                name = str(function_copy.get("name") or "").strip()
                if not name:
                    continue
                if not provider_name_re.match(name):
                    log.warning("Dropping provider-invalid tool schema name: %s", name)
                    continue
                if name in seen_tool_names:
                    log.warning("Dropping duplicate tool schema: %s", name)
                    continue
                seen_tool_names.add(name)
                function_copy["name"] = name
                function_copy["description"] = LLMClient._stringify_tool_description(
                    function_copy.get("description")
                )
                if not isinstance(function_copy.get("parameters"), dict):
                    function_copy["parameters"] = {"type": "object", "properties": {}}
                tool_copy["function"] = function_copy
            else:
                continue
            sanitized_tools.append(tool_copy)
        sanitized_tools.sort(key=lambda tool: str((tool.get("function") or {}).get("name") or ""))
        return sanitized_tools

    @staticmethod
    def _build_anthropic_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return None
        if tool_choice in {"required", "any"}:
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
        if isinstance(tool_choice, str):
            return {"type": "tool", "name": tool_choice}
        return None

    def _normalize_anthropic_response(
        self,
        resp_dict: Dict[str, Any],
        target: Dict[str, Any],
        prompt_cache_ttl: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        content_blocks = resp_dict.get("content") or []
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip()
            if block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })

        raw_usage = resp_dict.get("usage") or {}
        usage: Dict[str, Any] = {
            "prompt_tokens": int(raw_usage.get("input_tokens") or 0),
            "completion_tokens": int(raw_usage.get("output_tokens") or 0),
            "cached_tokens": int(raw_usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_creation_input_tokens") or 0),
            "provider": "anthropic",
            "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or ""),
        }
        if prompt_cache_ttl:
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            from ouroboros.pricing import estimate_cost

            estimated_cost = estimate_cost(
                usage["resolved_model"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["cached_tokens"],
                usage["cache_write_tokens"],
                usage.get("prompt_cache_ttl"),
            )
            if estimated_cost:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message, usage

    def _chat_anthropic(
        self,
        target: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        temperature: Optional[float] = None,
        no_proxy: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import requests

        del reasoning_effort  # Anthropic direct works without an extra effort payload here.

        system, anthropic_messages = self._build_anthropic_messages(messages)
        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        usage_model = str(target.get("usage_model") or target.get("resolved_model") or "")
        if temperature is not None:
            payload["temperature"] = temperature
        self._apply_rejected_param_cache(payload, usage_model)

        anthropic_tools = self._build_anthropic_tools(
            tools,
            cache_control=True,
        )
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            anthropic_tool_choice = self._build_anthropic_tool_choice(tool_choice)
            if anthropic_tool_choice:
                payload["tool_choice"] = anthropic_tool_choice
        prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
            payload.get("system"),
            payload.get("messages"),
            payload.get("tools"),
        )

        url = f"{str(target.get('base_url') or '').rstrip('/')}/messages"
        headers = {
            "x-api-key": str(target.get("api_key") or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        def _send(candidate: Dict[str, Any]):
            if no_proxy:
                # Build a session with proxy detection disabled for macOS fork-safety.
                # Use context manager (or explicit close) to avoid connection-pool leaks.
                with requests.Session() as session:
                    session.trust_env = False
                    sent = session.post(url, headers=headers, json=candidate, timeout=120)
            else:
                sent = requests.post(url, headers=headers, json=candidate, timeout=120)
            sent.raise_for_status()
            return sent

        try:
            response = _send(payload)
        except Exception as exc:
            retry_payload = self._retry_without_optional_sampling(payload, usage_model, exc)
            if retry_payload is None:
                raise
            response = _send(retry_payload)
        return self._normalize_anthropic_response(
            response.json(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )

    # ------------------------------------------------------------------
    # GigaChat (native `gigachat` library — NOT OpenAI-compatible)
    # ------------------------------------------------------------------
    def _get_gigachat_client(self, target: Dict[str, Any]):
        """Build (and cache) a GigaChat library client for the given target.

        Auth is whatever the env provides: an authorization key (``credentials``
        + ``scope``, OAuth) or ``user``/``password`` (basic auth). The library
        exchanges these for a short-lived access token and refreshes it
        automatically, so caching the client across calls is safe. Any other
        ``GIGACHAT_*`` setting present in the environment (e.g.
        ``GIGACHAT_PROFANITY_CHECK``) is picked up by the library itself.
        """
        credentials = str(target.get("api_key") or "")
        user = str(target.get("user") or "")
        password = str(target.get("password") or "")
        scope = str(target.get("scope") or "GIGACHAT_API_PERS")
        base_url = str(target.get("base_url") or "")
        verify = bool(target.get("verify_ssl_certs", True))
        cache_key = (credentials, user, password, scope, base_url, verify)

        client = self._gigachat_clients.get(cache_key)
        if client is None:
            try:
                from gigachat import GigaChat
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise RuntimeError(
                    "The 'gigachat' package is required to use gigachat:: models. "
                    "Install it with: pip install gigachat"
                ) from exc
            kwargs: Dict[str, Any] = {"scope": scope, "verify_ssl_certs": verify}
            if credentials:
                kwargs["credentials"] = credentials
            if user:
                kwargs["user"] = user
            if password:
                kwargs["password"] = password
            if base_url:
                kwargs["base_url"] = base_url
            client = GigaChat(**kwargs)
            self._gigachat_clients[cache_key] = client
        return client

    @staticmethod
    def _gigachat_text(content: Any) -> str:
        """Flatten OpenAI message content (str or list of blocks) to plain text.

        GigaChat messages carry a plain-string ``content``; multipart blocks and
        any ``cache_control`` markers are collapsed/dropped here.
        """
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content or "")

    @classmethod
    def _gigachat_function_result(cls, content: Any) -> str:
        """Return a function-result string that GigaChat accepts.

        GigaChat requires the ``function``-role message content to be a valid
        JSON document (it parses it server-side). Agent tool results are usually
        plain text (file contents, command output), so anything that isn't
        already valid JSON is wrapped as ``{"result": "<text>"}``.
        """
        text = cls._gigachat_text(content)
        try:
            json.loads(text)
            return text  # already valid JSON — pass through unchanged
        except Exception:
            return json.dumps({"result": text}, ensure_ascii=False)

    @classmethod
    def _gigachat_messages(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to GigaChat's message list.

        Differences handled here:
        - role ``tool`` (a tool result) → role ``function`` with the function
          ``name`` resolved from the originating assistant ``tool_call_id``.
        - assistant ``tool_calls`` (a list) → a single ``function_call`` object.
          GigaChat supports ONE function call per turn, so parallel tool calls
          are collapsed to the first one.
        """
        out: List[Dict[str, Any]] = []
        call_id_to_name: Dict[str, str] = {}
        last_function_name: Optional[str] = None

        for msg in messages:
            role = str(msg.get("role") or "")

            if role == "tool":
                name = (
                    call_id_to_name.get(str(msg.get("tool_call_id") or ""))
                    or last_function_name
                    or "function"
                )
                out.append({
                    "role": "function",
                    "name": name,
                    "content": cls._gigachat_function_result(msg.get("content")),
                })
                continue

            effective_role = role if role in ("system", "user", "assistant") else "user"
            # GigaChat requires the system message to be the FIRST message and
            # rejects any later one ("system message must be the first message").
            # The agent injects system-reminders mid-conversation, so demote any
            # non-leading system message to a user message (keeps its content and
            # recency, which matters for reminders).
            if effective_role == "system" and out:
                effective_role = "user"

            gmsg: Dict[str, Any] = {
                "role": effective_role,
                "content": cls._gigachat_text(msg.get("content")),
            }

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                # Record every id→name so following tool results resolve their
                # function name, but only the first call is sent to GigaChat.
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tcid = str(tc.get("id") or "")
                    tcname = str((tc.get("function") or {}).get("name") or "")
                    if tcid and tcname:
                        call_id_to_name[tcid] = tcname

                first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                fn = first.get("function") or {}
                name = str(fn.get("name") or "")
                args_raw = fn.get("arguments")
                arguments: Dict[str, Any] = {}
                if isinstance(args_raw, dict):
                    arguments = args_raw
                elif isinstance(args_raw, str) and args_raw.strip():
                    try:
                        arguments = json.loads(args_raw)
                    except Exception:
                        arguments = {}
                gmsg["function_call"] = {"name": name, "arguments": arguments}
                last_function_name = name

            out.append(gmsg)

        return out

    @staticmethod
    def _gigachat_sanitize_schema(node: Any) -> Any:
        """Make a JSON-Schema node acceptable to GigaChat's stricter validator.

        GigaChat rejects any ``"type": "object"`` node that lacks a ``properties``
        key with HTTP 422 ("Field is missing"), whereas OpenAI/JSON-Schema allow a
        free-form object. Recursively ensure every object node carries
        ``properties`` (default ``{}``), descending through ``properties`` values,
        array ``items``, ``additionalProperties``, and ``anyOf``/``oneOf``/``allOf``.
        ``cache_control`` markers are dropped wherever they appear.
        """
        if isinstance(node, list):
            return [LLMClient._gigachat_sanitize_schema(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "cache_control":
                continue
            if key == "properties" and isinstance(value, dict):
                out[key] = {
                    pk: LLMClient._gigachat_sanitize_schema(pv) for pk, pv in value.items()
                }
            elif key in ("items", "additionalProperties") and isinstance(value, (dict, list)):
                out[key] = LLMClient._gigachat_sanitize_schema(value)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                out[key] = [LLMClient._gigachat_sanitize_schema(v) for v in value]
            else:
                out[key] = value
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        return out

    @staticmethod
    def _gigachat_functions(
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI tool definitions to GigaChat ``functions`` entries."""
        functions: List[Dict[str, Any]] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if "function" in tool else tool
            fn = fn or {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            entry: Dict[str, Any] = {"name": name}
            if fn.get("description"):
                entry["description"] = str(fn["description"])
            params = fn.get("parameters")
            if isinstance(params, dict):
                entry["parameters"] = LLMClient._gigachat_sanitize_schema(params)
            functions.append(entry)
        return functions

    def _chat_gigachat(
        self,
        target: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        temperature: Optional[float] = None,
        no_proxy: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # The gigachat library owns its own httpx transport and proxy handling;
        # no_proxy (a macOS fork-safety flag for the OpenAI/requests paths) does
        # not apply here.
        del no_proxy

        client = self._get_gigachat_client(target)

        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": self._gigachat_messages(messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        functions = self._gigachat_functions(tools)
        if functions:
            payload["functions"] = functions
            # GigaChat accepts "auto"/"none" (or a specific {name}); it has no
            # strict "required", so anything else maps to "auto".
            payload["function_call"] = tool_choice if tool_choice in ("auto", "none") else "auto"

        effort = normalize_reasoning_effort(reasoning_effort)
        if effort in ("low", "medium", "high"):
            payload["reasoning_effort"] = effort

        completion = client.chat(payload)
        return self._normalize_gigachat_response(completion, target)

    def _normalize_gigachat_response(
        self,
        completion: Any,
        target: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a GigaChat ``ChatCompletion`` into (message, usage) dicts.

        A GigaChat ``function_call`` becomes a single OpenAI-style ``tool_calls``
        entry (arguments re-encoded as a JSON string). Cost is estimated from
        token counts via the local pricing table (GigaChat exposes no cost API).
        """
        choices = getattr(completion, "choices", None) or []
        first = choices[0] if choices else None
        gmsg = getattr(first, "message", None) if first is not None else None

        content = (getattr(gmsg, "content", "") or "") if gmsg is not None else ""
        message: Dict[str, Any] = {"role": "assistant", "content": content}

        function_call = getattr(gmsg, "function_call", None) if gmsg is not None else None
        if function_call is not None:
            name = getattr(function_call, "name", "") or ""
            arguments = getattr(function_call, "arguments", None)
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                args_str = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                args_str = "{}"
            message["tool_calls"] = [{
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }]
            # OpenAI convention: content is None when the turn is a tool call.
            if not content:
                message["content"] = None

        usage_obj = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0
        cached_tokens = int(getattr(usage_obj, "precached_prompt_tokens", 0) or 0) if usage_obj is not None else 0

        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cached_tokens,
            "provider": str(target.get("provider") or "gigachat"),
            "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or ""),
        }
        if prompt_tokens or completion_tokens:
            from ouroboros.pricing import estimate_cost

            estimated = estimate_cost(
                usage["resolved_model"], prompt_tokens, completion_tokens, cached_tokens, 0
            )
            if estimated:
                usage["cost"] = estimated
        usage.setdefault("cost", 0.0)

        return message, usage

    def _build_remote_kwargs(
        self,
        target: Dict[str, Any],
        messages: List[Dict[str, Any]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        temperature: Optional[float],
        tools: Optional[List[Dict[str, Any]]],
        skip_capability_fetch: bool = False,
    ) -> Dict[str, Any]:
        resolved_model = str(target.get("resolved_model") or "")
        token_limit_key = "max_tokens"
        if str(target.get("provider") or "") == "openai" and resolved_model.startswith("gpt-5"):
            token_limit_key = "max_completion_tokens"
        if not target.get("supports_openrouter_extensions"):
            # Non-OpenRouter providers do not accept cache_control.
            clean_messages = self._strip_openrouter_roundtrip_metadata(
                self._copy_messages_with_cache_policy(
                    messages,
                    allow_message_cache_control=False,
                    flatten_tool_content_blocks=True,
                )
            )
            kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "messages": clean_messages,
                token_limit_key: max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = [
                    {k: v for k, v in tool.items() if k != "cache_control"}
                    for tool in self._sanitize_chat_completion_tools(tools)
                ]
                kwargs["tool_choice"] = tool_choice
            self._apply_rejected_param_cache(kwargs, str(target.get("usage_model") or resolved_model))
            return kwargs

        effort = normalize_reasoning_effort(reasoning_effort)
        raw_return_reasoning = os.environ.get("OUROBOROS_RETURN_REASONING")
        return_reasoning = (
            True if raw_return_reasoning is None
            else str(raw_return_reasoning).strip().lower() not in _FALSE_LIKE_ENV_VALUES
        )
        cache_model = resolved_model.strip().lstrip("~")
        allow_message_cache = (
            cache_model.startswith("anthropic/")
            or cache_model.startswith("google/gemini-")
        )
        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": not return_reasoning},
        }

        if cache_model.startswith("anthropic/"):
            extra_body["provider"] = {
                "require_parameters": True,
            }

        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": self._copy_messages_with_cache_policy(
                messages,
                allow_message_cache_control=allow_message_cache,
                flatten_tool_content_blocks=not allow_message_cache,
            ),
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            prepared_tools = [
                {k: v for k, v in tool.items() if k != "cache_control"}
                for tool in self._sanitize_chat_completion_tools(tools)
            ]
            if prepared_tools and cache_model.startswith("anthropic/"):
                last_tool = {**prepared_tools[-1]}
                last_tool["cache_control"] = {"type": "ephemeral"}
                prepared_tools[-1] = last_tool
            kwargs["tools"] = prepared_tools
            kwargs["tool_choice"] = tool_choice

        # With require_parameters, unsupported params cause OpenRouter 404s.
        # Unknown capabilities mean no stripping.
        self._apply_rejected_param_cache(kwargs, resolved_model)
        if skip_capability_fetch:
            supported = None
        else:
            supported = self._get_supported_parameters(resolved_model)
        if supported is not None:
            for sampling_param in _OPTIONAL_SAMPLING_PARAMS:
                if sampling_param not in supported and sampling_param in kwargs:
                    log.debug(
                        "Model %s does not list %s in supported_parameters; stripping",
                        resolved_model, sampling_param,
                    )
                    kwargs.pop(sampling_param, None)
        return kwargs

    def _normalize_remote_response(
        self,
        resp_dict: Dict[str, Any],
        target: Dict[str, Any],
        skip_cost_fetch: bool = False,
        prompt_cache_ttl: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize an OpenAI-compatible response; skip_cost_fetch keeps no_proxy pure."""
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        if resp_dict.get("id") and "response_id" not in msg:
            msg["response_id"] = resp_dict["id"]

        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])
        # LM Studio MLX exposes prefix-cache hits only in stderr/logs, not
        # OpenAI-compatible usage; cached_tokens=0 is therefore expected.

        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (
                    prompt_details_for_write.get("cache_write_tokens")
                    or prompt_details_for_write.get("cache_creation_tokens")
                    or prompt_details_for_write.get("cache_creation_input_tokens")
                )
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        if target.get("supports_openrouter_extensions") and not skip_cost_fetch:
            if not usage.get("cost"):
                gen_id = resp_dict.get("id") or ""
                if gen_id:
                    cost = self._fetch_generation_cost(gen_id, target)
                    if cost is not None:
                        usage["cost"] = cost

        usage["provider"] = str(target.get("provider") or "openrouter")
        usage["resolved_model"] = str(target.get("usage_model") or target.get("resolved_model") or "")
        if prompt_cache_ttl and not usage.get("prompt_cache_ttl"):
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        if not usage.get("cost") and (usage.get("prompt_tokens") or usage.get("completion_tokens")):
            from ouroboros.pricing import estimate_cost

            estimated_cost = estimate_cost(
                usage["resolved_model"],
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                int(usage.get("cached_tokens") or 0),
                int(usage.get("cache_write_tokens") or 0),
                usage.get("prompt_cache_ttl"),
                allow_live_fetch=not skip_cost_fetch,
            )
            if estimated_cost:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True

        return msg, usage

    def _chat_remote(
        self,
        target: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        temperature: Optional[float] = None,
        no_proxy: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send remote chat; no_proxy uses a one-shot client and skips OS proxy lookup."""
        if target.get("provider") == "anthropic":
            return self._chat_anthropic(
                target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                no_proxy=no_proxy,
            )

        if target.get("provider") == "gigachat":
            return self._chat_gigachat(
                target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                no_proxy=no_proxy,
            )

        if no_proxy:
            _oa_client, _http_client = self._make_no_proxy_client(target)
            try:
                kwargs = self._build_remote_kwargs(
                    target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
                    skip_capability_fetch=True,
                )
                prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
                    kwargs.get("messages"),
                    kwargs.get("tools"),
                )
                try:
                    resp = _oa_client.chat.completions.create(**kwargs)
                except Exception as exc:
                    retry_kwargs = self._retry_without_optional_sampling(
                        kwargs,
                        str(target.get("usage_model") or target.get("resolved_model") or ""),
                        exc,
                    )
                    if retry_kwargs is None:
                        raise
                    resp = _oa_client.chat.completions.create(**retry_kwargs)
                # Skip cost fetch here; it would re-enter OS proxy lookup.
                return self._normalize_remote_response(
                    resp.model_dump(),
                    target,
                    skip_cost_fetch=True,
                    prompt_cache_ttl=prompt_cache_ttl,
                )
            finally:
                try:
                    _http_client.close()
                except Exception:
                    pass

        client = self._get_remote_client(target)
        kwargs = self._build_remote_kwargs(
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools
        )
        prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
            kwargs.get("messages"),
            kwargs.get("tools"),
        )
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            retry_kwargs = self._retry_without_optional_sampling(
                kwargs,
                str(target.get("usage_model") or target.get("resolved_model") or ""),
                exc,
            )
            if retry_kwargs is None:
                raise
            resp = client.chat.completions.create(**retry_kwargs)
        return self._normalize_remote_response(
            resp.model_dump(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = DEFAULT_LIGHT_MODEL,
        max_tokens: int = 32768,
        reasoning_effort: str = "none",
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a lightweight vision query; image dicts use url or base64+mime."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "google/gemini-3.5-flash")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "google/gemini-3.5-flash")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
