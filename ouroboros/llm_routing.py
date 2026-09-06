"""Provider target resolution, client construction and route affinity.

Which provider a model id belongs to, which credentials and base url that route
uses, which client object serves it (cached, proxy-free, async or local), and
which affinity key keeps repeat calls on one warm upstream — all of it is the
same question: where does this call go. The probe entry points live here because
they answer that question about a route without being a chat turn; their
transport is ``llm_probe``'s, so a probe never touches the chat retry, fallback
or capability-learning paths.
"""


from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from ouroboros.provider_models import (
    DEEPSEEK_BASE_URL,
    PROVIDER_PREFIXES,
    normalize_anthropic_model_id,
    normalize_model_identity,
    resolve_minimax_base_url,
)


_OR_PROVIDER_PRESETS = {
    # Same-model provider failover versus reproducible provider pinning.
    "resilience": {"allow_fallbacks": True},
    "repro": {"allow_fallbacks": False},
}


def _resolve_or_provider() -> Dict[str, Any]:
    """Resolve ``OUROBOROS_OR_PROVIDER`` (a preset name or a raw JSON object) into an
    OpenRouter ``provider`` routing dict. Empty/unset/invalid -> ``{}`` (no routing)."""
    raw = (os.environ.get("OUROBOROS_OR_PROVIDER") or "").strip()
    if not raw:
        return {}
    preset = _OR_PROVIDER_PRESETS.get(raw.lower())
    if preset is not None:
        return dict(preset)
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


class _ProviderRoutingMixin:
    """Provider targets, client factories, affinity keys and the window probe."""

    @staticmethod
    def _prompt_cache_identity(model_id: str, messages: List[Dict[str, Any]]) -> str:
        """Stable, credential-free affinity key for one policy prefix.

        Ouroboros' Main context places stable policy/governance in the first
        system text block and dynamic evidence last.  Hash only that stable
        prefix plus the normalized model identity, so changing task evidence
        does not fragment the provider cache while different policies cannot
        collide.  Routes without a leading system prefix simply opt out.
        """
        if not messages or str(messages[0].get("role") or "") != "system":
            return ""
        content = messages[0].get("content")
        stable_prefix = ""
        if isinstance(content, str):
            stable_prefix = content
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    stable_prefix = text
                    break
        if not stable_prefix.strip():
            return ""
        identity = normalize_model_identity(model_id) or str(model_id or "").strip()
        digest = hashlib.sha256(
            f"{identity}\0{stable_prefix}".encode("utf-8")
        ).hexdigest()[:32]
        return f"ouroboros-{digest}"

    @staticmethod
    def _explicit_cache_affinity_identity(model_id: str, cache_affinity: str) -> str:
        """Caller-declared session affinity: stable across rounds of one logical
        surface (e.g. ``plan_review:<task>``) so OpenRouter sticky routing keeps
        repeat calls on the same upstream and its prompt cache warm. The model
        identity is folded in so two models never share a session bucket; the
        caller key deliberately excludes slot ids so N same-model reviewer slots
        keep today's provider-concentration behavior."""
        affinity = str(cache_affinity or "").strip()
        if not affinity:
            return ""
        identity = normalize_model_identity(model_id) or str(model_id or "").strip()
        digest = hashlib.sha256(
            f"{identity}\0{affinity}".encode("utf-8")
        ).hexdigest()[:32]
        return f"ouroboros-session-{digest}"

    @classmethod
    def _openrouter_session_identity(
        cls,
        model_id: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Conversation-stable OpenRouter affinity, bounded well below 256 chars."""
        prefix_identity = cls._prompt_cache_identity(model_id, messages)
        if not prefix_identity:
            return ""
        first_user: Any = ""
        for message in messages:
            if str(message.get("role") or "") == "user":
                first_user = message.get("content")
                break
        serialized_user = json.dumps(
            first_user,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(
            f"{prefix_identity}\0{serialized_user}".encode("utf-8")
        ).hexdigest()[:32]
        return f"ouroboros-session-{digest}"

    @staticmethod
    def _parse_provider_model(model: str) -> Tuple[str, str]:
        model_name = str(model or "").strip()
        for prefix, provider in PROVIDER_PREFIXES:
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
        if provider == "minimax":
            return f"minimax/{resolved_model}"
        if provider == "deepseek":
            return f"deepseek/{resolved_model}"
        return f"openai-compatible/{resolved_model}"

    def _resolve_remote_target(
        self,
        model: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        explicit_settings = settings is not None

        def configured(key: str, default: Any = "") -> Any:
            if explicit_settings:
                return settings.get(key, default)  # type: ignore[union-attr]
            return os.environ.get(key, default)

        provider, resolved_model = self._parse_provider_model(model)
        usage_model = self._qualified_model_name(provider, resolved_model)

        if provider == "openai":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": configured("OPENAI_API_KEY", ""),
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
                "api_key": configured("ANTHROPIC_API_KEY", ""),
                "base_url": "https://api.anthropic.com/v1",
                "default_headers": {},
                "contract_headers": {"anthropic-version": "2023-06-01"},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "minimax":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": configured("MINIMAX_API_KEY", ""),
                "base_url": resolve_minimax_base_url(configured("MINIMAX_REGION", "")),
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "deepseek":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": configured("DEEPSEEK_API_KEY", ""),
                # One official endpoint; no owner-configurable base URL
                # (proxy/mirror setups belong to the openai-compatible slot).
                "base_url": DEEPSEEK_BASE_URL,
                "default_headers": {},
                # v4 thinks by default and carries reasoning_effort; the
                # canonical scale is projected onto its low/high/max enum in
                # _build_remote_kwargs. Tool-bearing requests MUST replay every
                # previous assistant turn's reasoning_content (v4-pro enforces
                # with a 400; "" is accepted for foreign turns — probed 2026-09-01).
                "requires_reasoning_echo": True,
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "cloudru":
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": configured("CLOUDRU_FOUNDATION_MODELS_API_KEY", ""),
                "base_url": (
                    configured("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "") or ""
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
            verify_raw = (configured("GIGACHAT_VERIFY_SSL_CERTS", "") or "").strip().lower()
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": configured("GIGACHAT_CREDENTIALS", ""),
                "user": (configured("GIGACHAT_USER", "") or "").strip(),
                "password": configured("GIGACHAT_PASSWORD", "") or "",
                "base_url": (
                    configured("GIGACHAT_BASE_URL", "") or ""
                ).strip() or "https://api.giga.chat/v1",
                "scope": (configured("GIGACHAT_SCOPE", "") or "").strip() or "GIGACHAT_API_PERS",
                "verify_ssl_certs": verify_raw not in ("0", "false", "no", "off"),
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        if provider == "openai-compatible":
            compatible_key = (configured("OPENAI_COMPATIBLE_API_KEY", "") or "").strip()
            compatible_base_url = (configured("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip()
            legacy_base_url = (configured("OPENAI_BASE_URL", "") or "").strip()
            legacy_key = (configured("OPENAI_API_KEY", "") or "").strip()
            # A request-local mapping is authoritative as a PAIR: when its
            # dedicated compatible endpoint is present, an explicitly empty
            # compatible key must not be rehydrated from the legacy OpenAI key.
            # Ordinary env-based chat keeps the historical per-field fallback.
            if explicit_settings and compatible_base_url:
                api_key = compatible_key
                base_url = compatible_base_url
            else:
                api_key = compatible_key or legacy_key
                base_url = compatible_base_url or legacy_base_url
            return {
                "provider": provider,
                "resolved_model": resolved_model,
                "usage_model": usage_model,
                "api_key": api_key,
                "base_url": base_url,
                "default_headers": {},
                "supports_openrouter_extensions": False,
                "supports_generation_cost": False,
            }

        current_api_key = configured("OPENROUTER_API_KEY", "") if explicit_settings else self._api_key_override
        if current_api_key is None:
            current_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return {
            "provider": "openrouter",
            "resolved_model": resolved_model,
            "usage_model": usage_model,
            "api_key": current_api_key,
            "base_url": "https://openrouter.ai/api/v1" if explicit_settings else self._base_url,
            "default_headers": dict(OPENROUTER_APP_HEADERS),
            "supports_openrouter_extensions": True,
            "supports_generation_cost": True,
        }

    def _get_client(self):
        target = self._resolve_remote_target("openrouter::")
        return self._get_remote_client(target)

    @staticmethod
    def _new_remote_client(target: Dict[str, Any]):
        # The keepalive transport carries SDK-equivalent pool limits (an
        # explicit transport ignores the Client-level limits); on proxy-routed
        # installs the helper returns None and SDK defaults keep proxy mounts.
        from openai import OpenAI

        from ouroboros.net_transport import keepalive_http_client

        kwargs: Dict[str, Any] = {
            "api_key": str(target.get("api_key") or ""),
            "max_retries": 0,
        }
        http_client = keepalive_http_client()
        if http_client is not None:
            kwargs["http_client"] = http_client
        base_url = str(target.get("base_url") or "")
        headers = dict(target.get("default_headers") or {})
        if base_url:
            kwargs["base_url"] = base_url
        if headers:
            kwargs["default_headers"] = headers
        return OpenAI(**kwargs)

    def _get_remote_client(self, target: Dict[str, Any]):
        base_url = str(target.get("base_url") or "")
        api_key = str(target.get("api_key") or "")
        headers = tuple(sorted(
            (str(k), str(v)) for k, v in dict(target.get("default_headers") or {}).items()
        ))
        cache_key = (str(target.get("provider") or ""), base_url, api_key, headers)
        if cache_key not in self._remote_clients:
            self._remote_clients[cache_key] = self._new_remote_client(target)
        return self._remote_clients[cache_key]

    def probe_oversized_context(
        self, model: str, content: str, *,
        base_url: str = "", max_output_tokens: int = 8, timeout: float = 20.0,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        from ouroboros.llm_probe import probe_oversized_context

        return probe_oversized_context(
            self, model, content, base_url=base_url,
            max_output_tokens=max_output_tokens, timeout=timeout, api_key=api_key,
        )

    def probe_provider_readiness(
        self,
        model: str,
        *,
        settings: Dict[str, Any],
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        from ouroboros.llm_probe import probe_provider_readiness

        return probe_provider_readiness(self, model, settings=settings, timeout=timeout)

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

            from ouroboros.net_transport import keepalive_http_client

            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "max_retries": 0,
            }
            http_client = keepalive_http_client(async_client=True)
            if http_client is not None:
                kwargs["http_client"] = http_client
            if base_url:
                kwargs["base_url"] = base_url
            if headers_dict:
                kwargs["default_headers"] = headers_dict
            client = AsyncOpenAI(**kwargs)
            self._async_remote_clients[cache_key] = client
        return client

    @staticmethod
    def _no_proxy_timeout(read_timeout: Optional[float] = None):
        import httpx
        from ouroboros.config import get_llm_transport_read_timeout_sec

        read_write = (
            float(read_timeout) if read_timeout and read_timeout > 0
            else get_llm_transport_read_timeout_sec()
        )
        return httpx.Timeout(connect=30.0, read=read_write, write=read_write, pool=30.0)

    @classmethod
    def _make_no_proxy_client(cls, target: Dict[str, Any], timeout: Optional[float] = None):
        from ouroboros.net_transport import make_no_proxy_client

        return make_no_proxy_client(target, cls._no_proxy_timeout(timeout))

    @classmethod
    def _make_no_proxy_async_client(cls, target: Dict[str, Any], timeout: Optional[float] = None):
        from ouroboros.net_transport import make_no_proxy_async_client

        return make_no_proxy_async_client(target, cls._no_proxy_timeout(timeout))
