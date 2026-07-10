"""LLM client for OpenRouter, direct providers, and optional local inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.provider_models import PROVIDER_PREFIXES, normalize_anthropic_model_id, normalize_model_identity
from ouroboros.utils import in_worker_process

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3.5-flash"
_FALSE_LIKE_ENV_VALUES = {"", "0", "false", "no", "off"}


def supports_message_cache_control(model: str) -> bool:
    """Providers whose OpenRouter route honors message-level cache_control breakpoints.

    Single source of truth for the prompt-cache family check (was an inline hardcoded
    ``startswith`` list at the request-build site).
    """
    m = str(model or "").strip().lstrip("~")
    return m.startswith("anthropic/") or m.startswith("google/gemini-")


def _reasoning_signature_portable_across_or_providers(model: str) -> bool:
    """Families whose replayed reasoning signatures SURVIVE an OpenRouter same-model
    cross-provider failover — so the ``allow_fallbacks=false`` continuity pin is
    unnecessary and would only defeat rate-limit resilience by stranding a turn on one
    throttled upstream when a healthy sibling endpoint could serve it.

    Verified live via a same-model replay probe (2026-06: generate reasoning forcing
    provider A, replay the assistant turn forcing each sibling provider B, observe HTTP
    200): Anthropic thinking-block signatures port across Anthropic / Bedrock / Vertex /
    Azure; Gemini reasoning ports across Google Vertex / AI-Studio; OpenAI
    encrypted-reasoning items port across OpenAI / Azure. Other families (e.g.
    ``z-ai/glm``, ``deepseek``) are UNVERIFIED and keep the conservative pin; the reactive
    ``_openrouter_signature_retry_kwargs`` 400 strip-and-retry is the safety net for every
    family if a cross-provider switch ever rejects a replayed signature."""
    m = str(model or "").strip().lstrip("~")
    return (
        m.startswith("anthropic/")
        or m.startswith("google/gemini-")
        or m.startswith("openai/")
    )


_OR_PROVIDER_PRESETS = {
    # Everyday resilience: fail over to another PROVIDER of the SAME model on a
    # rate-limit/5xx (the model — and its context window — is unchanged), while the
    # default sticky provider keeps the prompt cache warm. No throughput hopping.
    "resilience": {"allow_fallbacks": True},
    # Reproducibility (fixed-model benchmark runs): pin, no provider failover.
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
_OPTIONAL_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
# Droppable optional request params = sampling + structured-output + effort hint.
# response_format is request INTENT, not required semantics: every consumer keeps a
# text-parse fallback (e.g. the safety supervisor's bracket-scan). reasoning_effort
# is likewise a hint — a provider that names-and-rejects it (e.g. an older model
# refusing "none") gets the same one-shot strip-and-retry instead of failing the
# call outright (review round 6: a rejected safety call would fail CLOSED and
# block benign commands).
# output_config / thinking are the Anthropic-direct adaptive-thinking effort carriers
# (v6.57.0); an OLDER Claude that rejects them gets the same strip-and-retry rather than
# a hard 400 — the effort control degrades gracefully to "no forced thinking".
_OPTIONAL_DROPPABLE_PARAMS = _OPTIONAL_SAMPLING_PARAMS + (
    "response_format", "reasoning_effort", "output_config", "thinking",
)


class LocalContextTooLargeError(RuntimeError):
    """Raised when a local model cannot fit context without silent truncation."""


def _estimate_message_chars(messages: List[Dict[str, Any]]) -> int:
    from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT

    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "") in ("image_url", "image"):
                    total += IMAGE_BLOCK_CHAR_EQUIVALENT
                    continue
                total += len(str(block.get("text", "")))
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
    # v6.57.0: the accepted set is the EFFORT_SCALE SSOT (config.py), so adding a
    # tier (e.g. `max`) happens in one place. Imported lazily to avoid a config
    # import cycle at module load.
    try:
        from ouroboros.config import EFFORT_SCALE as _SCALE
        allowed = set(_SCALE)
    except Exception:
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
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


def fetch_cloudru_pricing() -> Dict[str, Tuple[float, ...]]:
    """Fetch cloud.ru Foundation Models pricing as ``cloudru/<id>`` -> per-1M USD.

    cloud.ru's ``GET /v1/models`` returns per-model ``metadata`` with token costs
    (``prompt_tokens_cost``, ``generated_tokens_cost``, ``cache_read_tokens_cost``,
    ``cache_write_tokens_cost``) in RUB per 1M tokens — i.e. the real resale price
    the owner pays. We convert to USD via ``OUROBOROS_RUB_USD_RATE`` so the catalog
    is the SSOT for ALL cloud.ru models (no hardcoded per-model table). Models with
    ``is_billable`` false/None are free → no row (estimate_cost returns 0). Returns
    {} when no cloud.ru key is configured or the fetch fails (caller falls back to
    the static table). Tuples are ``(input, cached_read, cache_write, output)``."""
    import logging
    log = logging.getLogger("ouroboros.llm")

    api_key = (os.environ.get("CLOUDRU_FOUNDATION_MODELS_API_KEY", "") or "").strip()
    if not api_key:
        return {}
    try:
        import requests
    except ImportError:
        return {}

    base_url = (
        os.environ.get("CLOUDRU_FOUNDATION_MODELS_BASE_URL", "") or ""
    ).strip() or "https://foundation-models.api.cloud.ru/v1"
    try:
        rate = float(os.environ.get("OUROBOROS_RUB_USD_RATE", "") or 95.0)
    except (TypeError, ValueError):
        rate = 95.0
    if rate <= 0:
        rate = 95.0

    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = resp.json().get("data", []) or []

        def _rub_per_1m_to_usd(value: Any) -> Optional[float]:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if num < 0:  # cloud.ru uses -1 for "n/a" (e.g. embedding output)
                return None
            return round(num / rate, 6)

        pricing_dict: Dict[str, Tuple[float, ...]] = {}
        for model in models:
            model_id = str(model.get("id") or "").strip()
            meta = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
            if not model_id or not meta or not meta.get("is_billable"):
                continue
            prompt_price = _rub_per_1m_to_usd(meta.get("prompt_tokens_cost"))
            output_price = _rub_per_1m_to_usd(meta.get("generated_tokens_cost"))
            if prompt_price is None or output_price is None:
                continue
            cached_price = _rub_per_1m_to_usd(meta.get("cache_read_tokens_cost"))
            cache_write_price = _rub_per_1m_to_usd(meta.get("cache_write_tokens_cost"))
            row = (
                prompt_price,
                cached_price if cached_price is not None else prompt_price,
                cache_write_price if cache_write_price is not None else prompt_price,
                output_price,
            )
            pricing_dict[f"cloudru/{model_id}"] = row
            pricing_dict[f"cloudru::{model_id}"] = row

        log.info(f"Fetched pricing for {len(pricing_dict) // 2} models from cloud.ru")
        return pricing_dict
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch cloud.ru pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Routes calls to OpenRouter or a local llama-cpp-python server."""

    # Missing capabilities mean "unknown": keep kwargs instead of stripping them.
    _SUPPORTED_PARAMS_CACHE: Dict[str, set] = {}
    _SUPPORTED_PARAMS_FETCHED: bool = False
    # Did the one-shot /models fetch actually reach OpenRouter (HTTP 200 + parse)?
    # Distinguishes a provider OUTAGE from a route with no metadata, so Capability
    # Evidence can mark STATUS_FAILED (transient) vs STATUS_UNPROBEABLE (v6.33.0 P4).
    _CAPABILITIES_FETCH_OK: bool = False
    # OpenRouter-reported context window per model id (provider_metadata evidence).
    _CONTEXT_LENGTH_CACHE: Dict[str, int] = {}
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
        cls._CAPABILITIES_FETCH_OK = False  # set True only on a clean 200 + parse
        try:
            import requests
            # 5s, not 15s: this fetch is on the synchronous capability-probe path
            # behind the max-context-mode gate (settings save / max toggle). A slow
            # probe must fail-closed quickly (-> window unknown -> max blocked with
            # the owner-ack escape), never hang the save (v6.33.0 WS4 timing budget).
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=5,
            )
            if resp.status_code != 200:
                log.debug(
                    "OpenRouter /models returned %d; supported_parameters cache empty",
                    resp.status_code,
                )
                return
            from ouroboros.provider_models import update_vision_overlay

            for m in resp.json().get("data", []) or []:
                mid = m.get("id") or ""
                sp = m.get("supported_parameters")
                if mid and isinstance(sp, list) and sp:
                    cls._SUPPORTED_PARAMS_CACHE[mid] = set(sp)
                # Context window (provider_metadata Capability Evidence source).
                cl = m.get("context_length")
                if mid and isinstance(cl, (int, float)) and cl > 0:
                    cls._CONTEXT_LENGTH_CACHE[mid] = int(cl)
                # Vision overlay for supports_vision(): authoritative
                # input_modalities from the same /models payload.
                arch = m.get("architecture")
                if mid and isinstance(arch, dict):
                    modalities = arch.get("input_modalities")
                    if isinstance(modalities, list) and modalities:
                        update_vision_overlay(mid, "image" in modalities)
            cls._CAPABILITIES_FETCH_OK = True  # reached the provider and parsed it
        except Exception:
            log.debug("Failed to fetch OpenRouter model capabilities", exc_info=True)

    @classmethod
    def metadata_fetch_attempted_and_failed(cls) -> bool:
        """True when the one-shot OpenRouter /models fetch RAN but did not succeed
        (non-200 or transport error) — i.e. the provider was unreachable, distinct
        from 'not fetched yet'. Capability Evidence uses this to record STATUS_FAILED
        (a transient outage) instead of STATUS_UNPROBEABLE (no metadata source)."""
        return bool(cls._SUPPORTED_PARAMS_FETCHED and not cls._CAPABILITIES_FETCH_OK)

    @classmethod
    def _get_supported_parameters(cls, model_id: str) -> Optional[set]:
        """Return supported parameter names, or None when unknown/no stripping."""
        if not cls._SUPPORTED_PARAMS_FETCHED:
            cls._fetch_openrouter_capabilities()
        return cls._SUPPORTED_PARAMS_CACHE.get(model_id)

    @classmethod
    def openrouter_context_length(cls, model_id: str, *, allow_fetch: bool = True) -> int:
        """OpenRouter-reported context window (tokens) for a model id, else 0.

        provider_metadata Capability Evidence source. A successful /models fetch is
        cached and not repeated; pass allow_fetch=False to read only the existing
        cache (so a hot path never triggers a blocking /models call). On the
        capability-probe path (allow_fetch=True) a RE-fetch is allowed when the
        last fetch FAILED or the requested model is absent from the cache — so a
        transient outage isn't poisoned one-shot and a model picked while the
        provider is unreachable is correctly seen as a transport failure (and
        surfaced as a no-connection error), not silently 'unprobeable' (v6.33.0)."""
        mid = str(model_id or "")
        needs_fetch = (not cls._SUPPORTED_PARAMS_FETCHED) or (
            allow_fetch and (not cls._CAPABILITIES_FETCH_OK or mid not in cls._CONTEXT_LENGTH_CACHE)
        )
        if allow_fetch and needs_fetch:
            cls._fetch_openrouter_capabilities()
        return int(cls._CONTEXT_LENGTH_CACHE.get(mid, 0) or 0)

    @staticmethod
    def _parameter_rejection_error(exc: BaseException) -> bool:
        text = str(exc or "").lower()
        if not text:
            return False
        # OpenRouter rejects unsupported sampling params (with require_parameters)
        # as "No endpoints found that support the requested parameters: ...".
        # Require an explicit parameter signal so unrelated "no endpoints found"
        # errors (e.g. "...that support tool use") do not falsely match.
        # "reasoning" covers the OpenRouter NESTED carrier (extra_body.reasoning.*):
        # its rejections name "reasoning"/"reasoning.effort", never the top-level
        # "reasoning_effort" spelling (triad r6).
        _param_names = _OPTIONAL_DROPPABLE_PARAMS + ("reasoning",)
        if "no endpoints found" in text and (
            "requested parameter" in text
            or any(param in text for param in _param_names)
        ):
            return True
        if not any(param in text for param in _param_names):
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
                # Strict pydantic-style servers (Anthropic direct, vLLM/SGLang)
                # reject unknown fields as "Extra inputs are not permitted".
                "not permitted",
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

    # Sentinel for the OpenRouter NESTED effort carrier (extra_body.reasoning) in the
    # rejected-params cache — top-level pops cannot reach it (triad r6).
    _NESTED_REASONING_PARAM = "extra_body.reasoning"

    @classmethod
    def _apply_rejected_param_cache(cls, payload: Dict[str, Any], model_id: str) -> None:
        for param in cls._known_rejected_params(model_id):
            if param == cls._NESTED_REASONING_PARAM:
                eb = payload.get("extra_body")
                if isinstance(eb, dict):
                    eb.pop("reasoning", None)
                continue
            payload.pop(param, None)

    # v6.57.0 — learned reasoning-effort ceilings (Q7). In-process cache is the hot
    # path; a durable copy in capability_evidence.json (effort_ceilings namespace,
    # DATA_DIR-scoped) survives restart. Key = normalized model identity. Fail-open.
    _EFFORT_CEILING_CACHE: Dict[str, str] = {}
    _EFFORT_CEILING_LOADED: Set[str] = set()

    @classmethod
    def _effort_ceiling_for(cls, model_id: str) -> str:
        key = normalize_model_identity(model_id) or str(model_id or "")
        if not key:
            return ""
        if key in cls._EFFORT_CEILING_CACHE:
            return cls._EFFORT_CEILING_CACHE[key]
        if key in cls._EFFORT_CEILING_LOADED:
            return ""
        cls._EFFORT_CEILING_LOADED.add(key)
        try:
            from ouroboros.capability_evidence import get_effort_ceiling
            from ouroboros.config import DATA_DIR
            ceil = get_effort_ceiling(DATA_DIR, key)
            if ceil:
                cls._EFFORT_CEILING_CACHE[key] = ceil
            return ceil
        except Exception:
            return ""

    def _clamp_effort_for_model(self, model_id: str, effort: str) -> str:
        """Clamp a requested effort DOWN to the route's learned ceiling. Owner values
        are honored up to the real ceiling; an ACTUAL clamp is recorded on the client
        (thread-local) and merged into THIS call's usage dict by the chat methods, so
        the lowering lands in the durable llm_usage event as
        ``reasoning_effort_clamped={requested, applied, reason}`` (BIBLE P1 — never a
        silent lowering; adversarial r1 verified the disclosure was missing)."""
        if not hasattr(self, "_effort_clamp_tls"):
            self._effort_clamp_tls = threading.local()
        # Reset at every payload build: a note left by an ABORTED earlier attempt on
        # this thread must never mis-attribute a clamp to the next call.
        self._effort_clamp_tls.pending = None
        ceiling = self._effort_ceiling_for(model_id)
        if not ceiling:
            return effort
        from ouroboros.config import clamp_effort_to
        clamped = clamp_effort_to(effort, ceiling)
        if clamped != effort:
            self._effort_clamp_tls.pending = {
                "requested": effort,
                "applied": clamped,
                "reason": "learned_ceiling",
                "model": str(model_id or ""),
            }
        return clamped

    def _pop_effort_clamp_disclosure(self) -> Optional[Dict[str, Any]]:
        """The pending clamp record for THIS thread's in-flight call, if any."""
        tls = getattr(self, "_effort_clamp_tls", None)
        pending = getattr(tls, "pending", None) if tls is not None else None
        if tls is not None:
            tls.pending = None
        return pending if isinstance(pending, dict) else None

    @classmethod
    def _record_effort_ceiling(cls, model_id: str, current_effort: str) -> None:
        """A provider rejected `current_effort` for this model → the ceiling is one step
        below it. Record in-process + durably so subsequent calls clamp immediately.
        FLOOR (adversarial r1): never learn a ceiling below "low" — a rejection of the
        lowest thinking tiers means the CARRIER is unsupported (the existing drop-param
        retry handles that); recording "none"/"minimal" would permanently disable
        thinking for the whole route off one bad request."""
        from ouroboros.config import effort_one_step_down, effort_rank
        key = normalize_model_identity(model_id) or str(model_id or "")
        eff = str(current_effort or "").strip().lower()
        if not key or not eff:
            return
        ceiling = effort_one_step_down(eff)
        if effort_rank(ceiling) < effort_rank("low"):
            return
        prev = cls._EFFORT_CEILING_CACHE.get(key)
        # A lower ceiling always wins (never silently regain a rejected level).
        if prev and effort_rank(prev) <= effort_rank(ceiling):
            return
        cls._EFFORT_CEILING_CACHE[key] = ceiling
        try:
            from ouroboros.capability_evidence import record_effort_ceiling
            from ouroboros.config import DATA_DIR
            record_effort_ceiling(DATA_DIR, key, ceiling)
        except Exception:
            pass

    @staticmethod
    def _payload_effort(payload: Dict[str, Any]) -> str:
        """Read the effort carried by a request payload across provider shapes."""
        eff = str(payload.get("reasoning_effort") or "").strip().lower()
        if eff:
            return eff
        oc = payload.get("output_config")
        if isinstance(oc, dict) and str(oc.get("effort") or "").strip():
            return str(oc.get("effort")).strip().lower()
        eb = payload.get("extra_body")
        if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
            return str(eb["reasoning"].get("effort") or "").strip().lower()
        return ""

    @classmethod
    def _retry_without_optional_sampling(
        cls,
        payload: Dict[str, Any],
        model_id: str,
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        if not cls._parameter_rejection_error(exc):
            return None
        present = {param for param in _OPTIONAL_DROPPABLE_PARAMS if param in payload}
        _err_text = str(exc or "").lower()
        _effort_implicated = any(
            k in _err_text for k in ("reasoning_effort", "output_config", "thinking", "reasoning", "effort")
        )
        # The OpenRouter lane carries effort NESTED as extra_body.reasoning.effort —
        # invisible to the top-level scan above, so an xhigh/max rejection there
        # would neither retry nor learn a ceiling nor disclose (triad r6). Treat the
        # nested carrier as droppable when the error implicates effort.
        _eb = payload.get("extra_body")
        _nested_reasoning = isinstance(_eb, dict) and isinstance(_eb.get("reasoning"), dict)
        if _nested_reasoning and _effort_implicated:
            present.add(cls._NESTED_REASONING_PARAM)
        if not present:
            return None
        # v6.57.0: if the error SPECIFICALLY implicates an effort carrier, learn the
        # route's ceiling as one step below the requested effort so the NEXT call clamps
        # immediately (converges over calls; this call still degrades by dropping the
        # carrier). The text check is required: a GENERIC parameter rejection (e.g. a
        # temperature-only refusal) must NOT teach a phantom effort ceiling.
        if (
            present & {"reasoning_effort", "output_config", "thinking", cls._NESTED_REASONING_PARAM}
            and _effort_implicated
        ):
            cls._record_effort_ceiling(model_id, cls._payload_effort(payload))
        cls._remember_rejected_params(model_id, present)
        retry_payload = copy.deepcopy(payload)
        for param in present:
            if param == cls._NESTED_REASONING_PARAM:
                # Remove ONLY the nested reasoning carrier; provider routing and any
                # other extra_body keys must survive the retry.
                _retry_eb = retry_payload.get("extra_body")
                if isinstance(_retry_eb, dict):
                    _retry_eb.pop("reasoning", None)
                continue
            retry_payload.pop(param, None)
        log.warning(
            "Retrying %s without optional request parameter(s): %s",
            model_id or "(unknown model)",
            ", ".join(sorted(present)),
        )
        return retry_payload

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

    def probe_oversized_context(
        self, model: str, content: str, *,
        base_url: str = "", max_output_tokens: int = 8, timeout: float = 20.0,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Capability probe: send ONE deliberately over-window request on the model's
        OpenAI-compatible route and report the RAW outcome for window classification.

        This is a capability check, NOT a chat turn: it deliberately bypasses the
        chat/usage/observability path (a probe must not pollute task usage or count as
        an LLM round) and NEVER raises. The expected free case is a 4xx pre-inference
        reject whose body carries the limit; a rare 200-accept returns the echo +
        prompt_tokens (the caller treats it as possibly-paid -> owner-ack, never a
        silent confirm). When an explicit ``base_url`` is given (Settings save/toggle
        passes the route being fingerprinted) it overrides the env-resolved one so a
        route change verifies the NEW endpoint. Returns
        ``{ok, status_code, body, echoed_text, usage_prompt}``.
        """
        try:
            target = self._resolve_remote_target(model)
            if str(base_url or "").strip():
                target = {**target, "base_url": str(base_url).strip()}
            if api_key is not None:
                target = {**target, "api_key": api_key}
            oai = self._get_remote_client(target)
            # resolved_model is the provider REQUEST model ("gpt-5.5"), not the
            # slash-qualified usage/tracking name the API would reject.
            resolved_model = str(target.get("resolved_model") or model.split("::")[-1])
            provider = str(target.get("provider") or "")
        except Exception as exc:  # pragma: no cover - setup failure -> fail-closed
            return {"ok": False, "status_code": None, "body": f"probe setup failed: {type(exc).__name__}",
                    "echoed_text": "", "usage_prompt": 0}
        # Direct OpenAI GPT-5/o-series reject ``max_tokens`` and require
        # ``max_completion_tokens``; other OpenAI-compatible stacks take max_tokens.
        cap = {"max_completion_tokens": max_output_tokens} if provider == "openai" else {"max_tokens": max_output_tokens}
        try:
            resp = oai.with_options(timeout=timeout).chat.completions.create(
                model=resolved_model, messages=[{"role": "user", "content": content}], temperature=0, **cap,
            )
            echoed, usage_prompt = "", 0
            try:
                echoed = str(resp.choices[0].message.content or "")
                usage_prompt = int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
            except Exception:
                pass
            return {"ok": True, "status_code": 200, "body": "", "echoed_text": echoed, "usage_prompt": usage_prompt}
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            body = str(getattr(exc, "message", "") or getattr(exc, "body", "") or str(exc))
            return {"ok": False, "status_code": status if isinstance(status, int) else None,
                    "body": body, "echoed_text": "", "usage_prompt": 0}

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
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(
            trust_env=False,
            mounts={},
            timeout=cls._no_proxy_timeout(timeout),
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
    def _make_no_proxy_async_client(cls, target: Dict[str, Any], timeout: Optional[float] = None):
        import httpx
        from openai import AsyncOpenAI

        http_client = httpx.AsyncClient(
            trust_env=False,
            mounts={},
            timeout=cls._no_proxy_timeout(timeout),
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
                        # Anthropic 400s on cache_control set for an EMPTY text block;
                        # only cache a text block that actually has text (image/tool
                        # blocks keep their cache_control). Pure removal of an invalid
                        # cache_control — never rewrites content, so all lanes are safe.
                        empty_text = (
                            block.get("type") == "text"
                            and not str(block.get("text") or "").strip()
                        )
                        if (allow_message_cache_control
                                and isinstance(block.get("cache_control"), dict)
                                and not empty_text):
                            block["cache_control"] = {"type": "ephemeral"}
                        else:
                            block.pop("cache_control", None)
                        # Internal metadata (image eviction captions/paths)
                        # never leaves the process — strict providers 400 on
                        # unknown content-block fields.
                        for key in [k for k in block if str(k).startswith("_")]:
                            block.pop(key, None)
        return cleaned

    # Provider-private reasoning CONTENT blocks (Anthropic/Gemini-via-OpenRouter
    # shape: content:[{type:"thinking"|"reasoning", signature:...}]) carry a
    # signature that only the PRODUCING upstream family can validate. Replaying
    # them to another family is the source of the 400 "Invalid `signature` in
    # `thinking` block" fallback death.
    _REASONING_CONTENT_BLOCK_TYPES = frozenset({"thinking", "reasoning", "redacted_thinking"})

    @classmethod
    def _strip_openrouter_roundtrip_metadata(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip provider-private reasoning round-trip artifacts that a DIFFERENT
        upstream family rejects: assistant-level ``reasoning``/``reasoning_details``/
        ``reasoning_content``/``response_id`` keys AND ``thinking``/``reasoning``
        CONTENT blocks (plus any stray ``signature`` on other blocks). Returns a
        deep copy; the canonical transcript is untouched.

        ``reasoning_content`` is the OpenAI-compatible direct-provider field name
        (GLM / Z.AI / cloud.ru Foundation Models, legacy vLLM) — distinct from the
        OpenRouter/Anthropic ``reasoning``/``reasoning_details`` shapes. Strict
        OpenAI-compatible servers (vLLM/SGLang) reject an echoed ``reasoning_content``
        with HTTP 400 ``Extra inputs are not permitted``, so it must be scrubbed on
        the cloudru / openai-compatible / local lanes too."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            msg.pop("reasoning", None)
            msg.pop("reasoning_details", None)
            msg.pop("reasoning_content", None)
            msg.pop("response_id", None)
            content = msg.get("content")
            if isinstance(content, list):
                kept: List[Any] = []
                for block in content:
                    if isinstance(block, dict):
                        btype = str(block.get("type") or "").strip().lower()
                        if btype in cls._REASONING_CONTENT_BLOCK_TYPES:
                            continue
                        block.pop("signature", None)
                    kept.append(block)
                msg["content"] = kept
        return cleaned

    @staticmethod
    def _replace_image_blocks_with_placeholder(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace image content-blocks with an explicit text placeholder for a
        model that has NO native vision — a raw ``image_url`` sent to a blind model
        is silently ignored or 404s. Mirrors the local llama.cpp and GigaChat lanes.
        Returns a deep copy; the canonical transcript is untouched."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for idx, block in enumerate(content):
                if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                    caption = str(block.get("_caption") or "").strip()
                    suffix = f" — {caption}" if caption else ""
                    content[idx] = {"type": "text", "text": f"[image omitted: model has no vision{suffix}]"}
        return cleaned

    @staticmethod
    def _content_with_system_notice_marker(content: Any) -> Any:
        marker = "[SYSTEM NOTICE]\n"
        if isinstance(content, list):
            out = copy.deepcopy(content)
            if out and isinstance(out[0], dict) and str(out[0].get("type") or "") in {"text", "input_text", "output_text"}:
                out[0]["text"] = marker + str(out[0].get("text") or "")
                return out
            return [{"type": "text", "text": marker}] + out
        return marker + str(content or "")

    @staticmethod
    def _is_deferrable_image_user_turn(msg: Dict[str, Any]) -> bool:
        """True for a USER message whose content carries an image block but NO tool_result
        block and NO tool_call_id — i.e. a mid-round injected image (view_image /
        native screenshot) that must not split an assistant tool_use from its matching
        tool_result. A user turn that IS a tool answer (Anthropic-style tool_result content
        block, or an OpenAI tool message) is never deferred (the negative guard)."""
        if str(msg.get("role") or "").strip().lower() != "user":
            return False
        if msg.get("tool_call_id"):
            return False
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        has_image = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
            if btype == "tool_result":
                return False  # this user turn answers a tool call — never defer it
            if btype in {"image_url", "image"}:
                has_image = True
        return has_image

    @classmethod
    def _normalize_system_message_placement(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Demote runtime system notices after conversation start.

        Providers with strict chat templates require system messages to appear
        only before the first user/assistant/tool turn. Late notices are runtime
        reminders, so they keep recency as user notices. If a notice appears
        between an assistant tool-call message and its tool results, it is
        buffered until after the adjacent tool-result block.

        The same buffer also defers a mid-round image-bearing USER turn (P4a):
        view_image / native-screenshot injection can append a user(image) message
        between an assistant tool_use and its tool_result, which violates every
        provider's tool-call adjacency contract. Buffering it (then flushing after
        the window closes) keeps the tool_result adjacent to its tool_use. This is
        the single send-time chokepoint every provider builder funnels through, so
        the fix covers Anthropic/OpenAI/Gemini/GigaChat at once (Bible P2/P7).
        """
        out: List[Dict[str, Any]] = []
        buffered_notices: List[Dict[str, Any]] = []
        seen_non_system = False
        awaiting_tool_results = False

        def flush_buffered() -> None:
            nonlocal buffered_notices
            if buffered_notices:
                out.extend(buffered_notices)
                buffered_notices = []

        for original in messages:
            msg = copy.deepcopy(original)
            role = str(msg.get("role") or "").strip().lower()

            # P4a: defer an image-bearing user turn that lands inside an open
            # tool_use↔tool_result window — BEFORE the generic clear below, so it is
            # buffered (kept in order with any demoted system notice) rather than
            # inserted between the tool_calls and their results.
            if awaiting_tool_results and cls._is_deferrable_image_user_turn(msg):
                buffered_notices.append(msg)
                continue

            if awaiting_tool_results and role not in {"tool", "system"}:
                awaiting_tool_results = False
                flush_buffered()

            if role == "system" and seen_non_system:
                msg["role"] = "user"
                msg["content"] = cls._content_with_system_notice_marker(msg.get("content"))
                if awaiting_tool_results:
                    buffered_notices.append(msg)
                else:
                    out.append(msg)
                continue

            out.append(msg)
            if role != "system":
                seen_non_system = True
            if role == "assistant" and msg.get("tool_calls"):
                awaiting_tool_results = True

        flush_buffered()
        return out

    @staticmethod
    def _has_openrouter_reasoning_details(messages: List[Dict[str, Any]]) -> bool:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("reasoning_details"):
                return True
        return False

    @classmethod
    def _has_replayed_reasoning_metadata(cls, messages: List[Dict[str, Any]]) -> bool:
        """True if the transcript carries provider-private reasoning artifacts that
        a DIFFERENT upstream family cannot validate: assistant ``reasoning``/
        ``reasoning_details``/``reasoning_content``/``response_id`` keys, or
        ``thinking``/``reasoning`` CONTENT blocks (or a stray ``signature`` on a
        content block). Broader than ``_has_openrouter_reasoning_details`` (which
        only sees the top-level ``reasoning_details`` field)."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if (
                msg.get("reasoning")
                or msg.get("reasoning_details")
                or msg.get("reasoning_content")
                or msg.get("response_id")
            ):
                return True
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = str(block.get("type") or "").strip().lower()
                    if btype in cls._REASONING_CONTENT_BLOCK_TYPES or block.get("signature"):
                        return True
        return False

    @staticmethod
    def _model_family(model: Any) -> str:
        """The upstream provider FAMILY of a model id — the part before the first
        '/' (``z-ai/glm-5.2`` -> ``z-ai``; ``anthropic/claude-…`` -> ``anthropic``).
        This is the boundary that matters for reasoning-signature validity: GLM and
        Claude both transit OpenRouter, so ``provider=='openrouter'`` is too coarse —
        the FAMILY produces (and alone can validate) a thinking-block signature."""
        norm = (normalize_model_identity(str(model or "")) or str(model or "")).strip().lower().lstrip("~")
        if "/" in norm:
            return norm.split("/", 1)[0]
        return norm

    @staticmethod
    def _is_http_status(exc: Exception, code: int) -> bool:
        """Structural HTTP-status check on a provider exception (``status_code``
        attribute; falls back to the OpenAI-SDK ``Error code: NNN`` message shape).
        Used instead of error-string matching so the recovery covers every provider
        phrasing of the same status class."""
        sc = getattr(exc, "status_code", None)
        if sc is not None:
            try:
                return int(sc) == int(code)
            except (TypeError, ValueError):
                pass
        # No status_code attr (non-SDK exceptions): match the code only as a
        # STATUS token — leading, or after error/status/http labels — not any bare
        # number, so a token count or id with "400" in it can't false-trigger.
        text = str(exc).strip().lower()
        return bool(re.search(rf"(?:^|error code:?\s*|status(?:[ _]code)?:?\s*|http[\s:]*){int(code)}\b", text))

    def _openrouter_signature_retry_kwargs(
        self,
        target: Dict[str, Any],
        kwargs: Dict[str, Any],
        exc: Exception,
    ) -> Optional[Dict[str, Any]]:
        """Structural recovery for provider 400s caused by replaying reasoning
        metadata: when the request CARRIED replayed reasoning artifacts AND the
        provider returned 400, strip the artifacts and retry the SAME model once.
        The trigger is structural (request shape + 400 status), NOT an error-string
        allowlist — so every provider phrasing of this failure class is covered.
        ``_reroute_same_model_kwargs`` returns None when no reasoning was present,
        so a genuine (non-reasoning) 400 still propagates unchanged."""
        if not target.get("supports_openrouter_extensions"):
            return None
        if not self._is_http_status(exc, 400):
            return None
        return self._reroute_same_model_kwargs(target, kwargs)

    def _reroute_same_model_kwargs(
        self,
        target: Dict[str, Any],
        kwargs: Dict[str, Any],
        *,
        allow_portable_reasoning: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Same-model reroute: strip replayed reasoning metadata and drop the
        provider pin (``allow_fallbacks=false``, set only to preserve reasoning
        continuity) so OpenRouter can route to a HEALTHY endpoint of the SAME
        model. Shared by the 400 signature-rejection path and the transient
        200-body provider-error path. Returns None when no replayed reasoning is
        present (nothing to strip / no continuity pin to drop — default routing can
        already fall back across endpoints). NEVER switches model — only endpoint.

        ``allow_portable_reasoning`` (set ONLY by the transient body-error path): for a
        family whose reasoning signature is cross-provider portable
        (``_reasoning_signature_portable_across_or_providers``) the replayed signature
        survives the same-model sibling-provider switch, so PRESERVE it (retry the same
        payload and let OpenRouter route to a healthy endpoint) rather than needlessly
        dropping continuity on the very rate-limit path the failover exists for. The 400
        signature-REJECTION path never sets this: a 400 means the signature WAS rejected,
        so it must strip regardless of family."""
        if not target.get("supports_openrouter_extensions"):
            return None
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not self._has_replayed_reasoning_metadata(messages):
            return None
        if allow_portable_reasoning and _reasoning_signature_portable_across_or_providers(kwargs.get("model")):
            return copy.deepcopy(kwargs)
        retry_kwargs = copy.deepcopy(kwargs)
        retry_kwargs["messages"] = self._strip_openrouter_roundtrip_metadata(messages)
        if not self._has_replayed_reasoning_metadata(retry_kwargs["messages"]):
            extra_body = retry_kwargs.get("extra_body")
            provider = extra_body.get("provider") if isinstance(extra_body, dict) else None
            if isinstance(provider, dict):
                provider.pop("allow_fallbacks", None)
                if not provider:
                    extra_body.pop("provider", None)
                if not extra_body:
                    retry_kwargs.pop("extra_body", None)
        return retry_kwargs

    @classmethod
    def sanitize_reasoning_on_model_switch(
        cls,
        messages: List[Dict[str, Any]],
        from_model: Any,
        to_model: Any,
    ) -> List[Dict[str, Any]]:
        """SSOT for cross-family model switches (cross-model fallback, switch_model,
        per-task model override): when the TARGET model belongs to a DIFFERENT
        provider family than the SOURCE, strip provider-private reasoning artifacts
        the target cannot validate — this is what kills the GLM->Claude fallback
        with a 400 ``Invalid `signature` in `thinking` block``. Same family ->
        return ``messages`` unchanged (preserve reasoning continuity). On a switch
        returns a sanitized COPY; the canonical transcript is never mutated."""
        if cls._model_family(from_model) == cls._model_family(to_model):
            return messages
        return cls._strip_openrouter_roundtrip_metadata(messages)

    @staticmethod
    def _provider_body_error(resp_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """An OpenAI-compatible HTTP 200 whose body carries a top-level ``error``
        object instead of a usable completion. OpenRouter passes upstream
        provider errors and its own 429/5xx through the body with status 200; the
        OpenAI SDK builds these leniently, keeping ``error`` and ``choices=None``.
        Returns the error dict, else None (a real completion wins over a
        non-fatal error field)."""
        if not isinstance(resp_dict, dict):
            return None
        err = resp_dict.get("error")
        if not isinstance(err, dict):
            return None
        choices = resp_dict.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message") if isinstance(first, dict) else None
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                return None
        return err

    @staticmethod
    def _is_transient_body_error(err: Dict[str, Any]) -> bool:
        """Transient body-error = worth a same-model reroute/retry (rate limit,
        overload, upstream 5xx/timeout). Permanent client errors
        (auth/quota/bad-request) are not — they must surface unchanged."""
        try:
            code = int(err.get("code"))
        except (TypeError, ValueError):
            code = 0
        if code in (408, 409, 425, 429, 500, 502, 503, 504, 522, 524, 529):
            return True
        text = str(err.get("message") or "").lower()
        return any(
            marker in text
            for marker in (
                "rate limit", "too many requests", "overloaded", "temporarily",
                "timeout", "timed out", "unavailable", "try again", "capacity",
            )
        )

    def _reroute_kwargs_for_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """If an HTTP-200 response actually carries a TRANSIENT provider
        body-error, return same-model reroute kwargs (provider unpinned; reasoning
        continuity preserved for cross-provider-portable families, dropped
        otherwise); None when not applicable."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        err = self._provider_body_error(resp_dict)
        if not err or not self._is_transient_body_error(err):
            return None
        reroute = self._reroute_same_model_kwargs(
            target, kwargs, allow_portable_reasoning=True
        )
        if reroute is None:
            return None
        log.warning(
            "OpenRouter same-model reroute after transient provider body-error "
            "(code=%s); reasoning_continuity_%s",
            err.get("code"),
            "preserved"
            if _reasoning_signature_portable_across_or_providers(kwargs.get("model"))
            else "dropped",
        )
        return reroute

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
        timeout: Optional[float] = None,
        allow_server_web_search: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call returning (message, usage); no_proxy avoids macOS fork proxy crashes.

        ``response_format`` (e.g. ``{"type": "json_object"}``) is optional request
        intent on the OpenAI-compatible/OpenRouter lanes: local, Anthropic-native,
        and GigaChat routes ignore it, and a provider rejection strips it via the
        optional-parameter retry — callers must keep a text-parse fallback."""
        messages = self._normalize_system_message_placement(messages)
        if use_local:
            return self._chat_local(messages, tools, max_tokens, tool_choice, timeout=timeout)

        # Central worker policy: any LLM call from a worker process is fork-safe
        # by default (no system proxy lookup). This covers the main agent loop,
        # consolidator, post-task threads, and supervisor dedup without each
        # call site having to remember no_proxy=True.
        no_proxy = no_proxy or in_worker_process()
        target = self._resolve_remote_target(model)
        return self._chat_remote(
            target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
            no_proxy=no_proxy,
            timeout=timeout,
            allow_server_web_search=allow_server_web_search,
            response_format=response_format,
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
        timeout: Optional[float] = None,
        allow_server_web_search: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async remote chat; no_proxy keeps forked macOS workers off OS proxy APIs."""
        messages = self._normalize_system_message_placement(messages)
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
                timeout,
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
            _oa_client, _http_client = self._make_no_proxy_async_client(target, timeout=timeout)
            try:
                kwargs = self._build_remote_kwargs(
                    target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
                    skip_capability_fetch=True,
                    allow_server_web_search=allow_server_web_search,
                )
                prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
                    kwargs.get("messages"),
                    kwargs.get("tools"),
                )
                resp = await self._create_chat_completion_with_retries_async(
                    _oa_client.chat.completions.create,
                    kwargs,
                    target,
                )
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
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
            allow_server_web_search=allow_server_web_search,
        )
        if timeout and timeout > 0:
            # Cached clients are built without a timeout; honor the caller's
            # per-request timeout instead of silently using the SDK default.
            kwargs["timeout"] = float(timeout)
        prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
            kwargs.get("messages"),
            kwargs.get("tools"),
        )
        resp = await self._create_chat_completion_with_retries_async(
            client.chat.completions.create,
            kwargs,
            target,
        )
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
        timeout: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send a chat request to the local llama-cpp-python server."""
        client = self._get_local_client()

        messages = self._normalize_system_message_placement(messages)
        clean_messages = self._strip_openrouter_roundtrip_metadata(
            self._copy_messages_with_cache_policy(
                messages,
                allow_message_cache_control=False,
                flatten_tool_content_blocks=True,
            )
        )
        # Local llama.cpp lane has no vision: replace image blocks with an
        # explicit placeholder (a raw image_url block would be str()-flattened
        # into the prompt as base64 noise).
        for msg in clean_messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for idx, block in enumerate(content):
                if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                    content[idx] = {"type": "text", "text": "[image omitted: model has no vision]"}
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
        if timeout and timeout > 0:
            # Honor the caller's per-request timeout on the local lane too
            # (v6.54.3: the safety-supervisor timeout SSOT must bound every
            # route safety can use, not only the remote ones).
            kwargs["timeout"] = float(timeout)

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

    @staticmethod
    def _sanitize_anthropic_tool_result_content(content: Any) -> Any:
        """Anthropic rejects empty tool_result content (and 400s on cache_control set
        for an empty text block). Drop empty text blocks, KEEP non-empty / non-text
        (image/document/search) blocks, and substitute a single placeholder only when
        the whole tool result would otherwise be empty (scalar ``""`` or list ``[]``)."""
        placeholder = "(no tool output)"
        if isinstance(content, list):
            cleaned = [
                b for b in content
                if not (
                    isinstance(b, dict)
                    and str(b.get("type") or "") == "text"
                    and not str(b.get("text") or "").strip()
                )
            ]
            return cleaned if cleaned else placeholder
        text = "" if content is None else str(content)
        return text if text.strip() else placeholder

    def _build_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        messages = self._normalize_system_message_placement(messages)
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
                tool_result_content = self._sanitize_anthropic_tool_result_content(tool_result_content)
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
    def _openrouter_main_web_search_tool() -> Optional[Dict[str, Any]]:
        mode = str(os.environ.get("OUROBOROS_MAIN_WEB_SEARCH") or "off").strip().lower()
        if mode not in {"openrouter", "openrouter_server", "server", "on", "true", "1"}:
            return None
        engine = str(os.environ.get("OUROBOROS_MAIN_WEB_SEARCH_ENGINE") or "auto").strip() or "auto"
        parameters: Dict[str, Any] = {}
        if engine != "auto":
            parameters["engine"] = engine
        try:
            max_total = int(os.environ.get("OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS", "") or 0)
        except ValueError:
            max_total = 0
        if max_total > 0:
            parameters["max_total_results"] = max_total
        tool: Dict[str, Any] = {"type": "openrouter:web_search"}
        if parameters:
            tool["parameters"] = parameters
        return tool

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
        # v6.61.1 (Q7 disclosure): a learned-ceiling clamp on this call rides the usage
        # event — "requested xhigh → applied high (learned_ceiling)" is never silent.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        # Anthropic always returns stop_reason on success; surface it so the empty-
        # response classifier isn't blind on the direct lane (otherwise every direct
        # response looks like a finish_reason=null transient glitch).
        stop_reason = resp_dict.get("stop_reason")
        if stop_reason:
            message["stop_reason"] = str(stop_reason)
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
        timeout: Optional[float] = None,
        allow_server_web_search: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import requests

        system, anthropic_messages = self._build_anthropic_messages(messages)
        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        # v6.57.0 — direct Anthropic honors the configured reasoning effort instead of
        # silently dropping it (the old `del reasoning_effort` made the effort control a
        # no-op for direct Opus). Modern Claude (Opus 4.7+/Sonnet 5/Fable-5) uses ADAPTIVE
        # thinking + top-level `output_config.effort` (manual budget_tokens now 400s), with
        # levels low|medium|high|xhigh|max — a superset of our scale, so the normalized
        # value maps 1:1 (none→omit thinking = no forced reasoning). A model that rejects
        # `output_config`/`thinking` triggers the same param-rejection retry as other
        # optional params (learned + dropped), so an older Claude never hard-fails on this.
        _eff = self._clamp_effort_for_model(
            str(target.get("usage_model") or target.get("resolved_model") or ""),
            normalize_reasoning_effort(reasoning_effort),
        )
        if _eff and _eff != "none":
            payload["thinking"] = {"type": "adaptive"}
            # Anthropic's documented effort set is low|medium|high|xhigh|max — it has
            # no "minimal", so OUR lowest thinking tier maps to the provider's floor
            # (adversarial r1: the old identity mapping could send an out-of-range
            # value and poison the route's learned ceiling).
            payload["output_config"] = {"effort": "low" if _eff == "minimal" else _eff}
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
        request_timeout = float(timeout) if timeout and timeout > 0 else 120

        def _send(candidate: Dict[str, Any]):
            if no_proxy:
                # Build a session with proxy detection disabled for macOS fork-safety.
                # Use context manager (or explicit close) to avoid connection-pool leaks.
                with requests.Session() as session:
                    session.trust_env = False
                    sent = session.post(url, headers=headers, json=candidate, timeout=request_timeout)
            else:
                sent = requests.post(url, headers=headers, json=candidate, timeout=request_timeout)
            if sent.status_code >= 400:
                # Include the Anthropic error body: requests' bare HTTPError text
                # ("400 Client Error … for url …") carries no parameter names, so
                # the sampling-reject retry matcher could never fire on this path.
                body_preview = (sent.text or "")[:2000]
                raise requests.HTTPError(
                    f"{sent.status_code} {sent.reason} for url {sent.url}: {body_preview}",
                    response=sent,
                )
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
    def _get_gigachat_client(self, target: Dict[str, Any], timeout: Optional[float] = None):
        """Build (and cache) a GigaChat library client for the given target.

        Auth is whatever the env provides: an authorization key (``credentials``
        + ``scope``, OAuth) or ``user``/``password`` (basic auth). The library
        exchanges these for a short-lived access token and refreshes it
        automatically, so caching the client across calls is safe. Any other
        ``GIGACHAT_*`` setting present in the environment (e.g.
        ``GIGACHAT_PROFANITY_CHECK``) is picked up by the library itself.
        A caller-supplied per-request ``timeout`` becomes part of the cache key
        (the library takes it at construction), so the safety-supervisor timeout
        SSOT bounds this lane too (v6.54.3)."""
        credentials = str(target.get("api_key") or "")
        user = str(target.get("user") or "")
        password = str(target.get("password") or "")
        scope = str(target.get("scope") or "GIGACHAT_API_PERS")
        base_url = str(target.get("base_url") or "")
        verify = bool(target.get("verify_ssl_certs", True))
        timeout_key = float(timeout) if timeout and timeout > 0 else None
        cache_key = (credentials, user, password, scope, base_url, verify, timeout_key)

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
            if timeout_key is not None:
                kwargs["timeout"] = timeout_key
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
                    if str(block.get("type") or "") in ("image_url", "image"):
                        # Explicit placeholder instead of a silent drop: the
                        # model (and the transcript reader) must know an image
                        # was present but not deliverable on this lane.
                        caption = str(block.get("_caption") or "").strip()
                        parts.append(f"[image omitted: model has no vision{f' — {caption}' if caption else ''}]")
                        continue
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
        messages = cls._normalize_system_message_placement(messages)
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
        timeout: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # The gigachat library owns its own httpx transport and proxy handling;
        # no_proxy (a macOS fork-safety flag for the OpenAI/requests paths) does
        # not apply here.
        del no_proxy

        client = self._get_gigachat_client(target, timeout=timeout)

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

        # Current GigaChat-3 models can spend the full max_tokens budget on
        # hidden reasoning and return empty content/tool_calls when
        # reasoning_effort is sent. Keep the native path deterministic.

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
        allow_server_web_search: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        messages = self._normalize_system_message_placement(messages)
        resolved_model = str(target.get("resolved_model") or "")
        provider = str(target.get("provider") or "")
        # Blind-model image placeholder applies to BOTH the direct (OpenAI/OpenAI-
        # compatible/Cloud.ru) and OpenRouter lanes (C2.3): a model with no native
        # vision gets an explicit "[image omitted]" placeholder instead of raw image
        # blocks it would 404/ignore. Done BEFORE the provider-branch split so the
        # direct branch (which returns early below) is covered too — mirrors the
        # local/GigaChat lanes; the VLM tool lane already routes vision to a capable
        # slot. supports_vision() is a no-op for vision-capable models.
        from ouroboros.provider_models import supports_vision
        if not supports_vision(resolved_model):
            messages = self._replace_image_blocks_with_placeholder(messages)
        # OpenAI reasoning models (gpt-5*, o-series) reject legacy max_tokens
        # with a deterministic 400 — they require max_completion_tokens.
        openai_reasoning_model = provider == "openai" and resolved_model.startswith(
            ("gpt-5", "o1", "o3", "o4")
        )
        token_limit_key = "max_completion_tokens" if openai_reasoning_model else "max_tokens"
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
            if openai_reasoning_model:
                # Direct-OpenAI route honors the configured OUROBOROS_EFFORT_*
                # lanes instead of silently dropping them (OpenRouter parity).
                # v6.57.0: clamp to the route's learned ceiling (e.g. a model that
                # tops out at high never re-errors on a global xhigh — it clamps down).
                _oa_eff = self._clamp_effort_for_model(
                    str(target.get("usage_model") or resolved_model),
                    normalize_reasoning_effort(reasoning_effort),
                )
                kwargs["reasoning_effort"] = _oa_eff
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format:
                kwargs["response_format"] = dict(response_format)
            if tools:
                kwargs["tools"] = [
                    {k: v for k, v in tool.items() if k != "cache_control"}
                    for tool in self._sanitize_chat_completion_tools(tools)
                ]
                kwargs["tool_choice"] = tool_choice
            self._apply_rejected_param_cache(kwargs, str(target.get("usage_model") or resolved_model))
            return kwargs

        effort = self._clamp_effort_for_model(
            str(target.get("usage_model") or resolved_model),
            normalize_reasoning_effort(reasoning_effort),
        )
        raw_return_reasoning = os.environ.get("OUROBOROS_RETURN_REASONING")
        return_reasoning = (
            True if raw_return_reasoning is None
            else str(raw_return_reasoning).strip().lower() not in _FALSE_LIKE_ENV_VALUES
        )
        cache_model = resolved_model.strip().lstrip("~")
        allow_message_cache = supports_message_cache_control(resolved_model)
        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": not return_reasoning},
        }

        if cache_model.startswith("anthropic/"):
            extra_body["provider"] = {
                "require_parameters": True,
            }
        # Replayed reasoning is endpoint-bound ONLY for families whose thought-block
        # signatures do not survive a same-model cross-provider switch. Anthropic, Gemini
        # and OpenAI reasoning signatures ARE cross-provider portable on OpenRouter
        # (Anthropic across Anthropic/Bedrock/Vertex/Azure; Gemini across Vertex/AI-Studio;
        # OpenAI encrypted items across OpenAI/Azure — live same-model replay probe, 2026-06:
        # each minted signature validated 200 on its sibling providers), so they must stay
        # failover-eligible. Pinning them would defeat OpenRouter's same-model provider
        # resilience and surface one upstream's rate-limit when a healthy sibling endpoint
        # could serve the turn. OpenRouter routing is sticky (the same provider serves the
        # happy path), so the prompt cache stays warm on the primary and only a real
        # outage triggers the cross-provider failover — no throughput hopping. Unverified
        # families (e.g. z-ai/glm, deepseek) keep the conservative pin; the reactive 400
        # strip-and-retry (_openrouter_signature_retry_kwargs) is the safety net for all.
        # The trigger is the BROAD replay-artifact contract (_has_replayed_reasoning_metadata
        # — assistant reasoning/reasoning_content/response_id OR a signed reasoning/thinking
        # CONTENT block), matching the reactive strip path, so an unverified signed block
        # cannot slip past the pin via a non-`reasoning_details` artifact.
        if self._has_replayed_reasoning_metadata(messages) and not _reasoning_signature_portable_across_or_providers(cache_model):
            provider_body = extra_body.setdefault("provider", {})
            if isinstance(provider_body, dict):
                provider_body["allow_fallbacks"] = False
        # Owner-configured OpenRouter provider routing (resilience/repro). Gap-merge:
        # NEVER override the anthropic require_parameters pin or the (unverified-family)
        # reasoning-continuity allow_fallbacks=False pin set above. Affects same-model
        # provider routing only — it never changes the MODEL, so the P3 reviewer context
        # floor is untouched.
        _or_provider = _resolve_or_provider()
        if _or_provider:
            provider_body = extra_body.setdefault("provider", {})
            if isinstance(provider_body, dict):
                for _k, _v in _or_provider.items():
                    if _k == "require_parameters" and provider_body.get("require_parameters"):
                        continue
                    if _k == "allow_fallbacks" and provider_body.get("allow_fallbacks") is False:
                        continue
                    provider_body[_k] = _v

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
        if response_format:
            kwargs["response_format"] = dict(response_format)
        server_web_tool = (
            self._openrouter_main_web_search_tool()
            if (tools and allow_server_web_search)
            else None
        )
        if tools or server_web_tool:
            prepared_tools = [
                {k: v for k, v in tool.items() if k != "cache_control"}
                for tool in self._sanitize_chat_completion_tools(tools)
            ]
            if server_web_tool:
                prepared_tools.append(server_web_tool)
            if prepared_tools and cache_model.startswith("anthropic/"):
                for idx in range(len(prepared_tools) - 1, -1, -1):
                    if isinstance(prepared_tools[idx].get("function"), dict):
                        last_tool = {**prepared_tools[idx]}
                        last_tool["cache_control"] = {"type": "ephemeral"}
                        prepared_tools[idx] = last_tool
                        break
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
            for optional_param in _OPTIONAL_DROPPABLE_PARAMS:
                if optional_param not in supported and optional_param in kwargs:
                    log.debug(
                        "Model %s does not list %s in supported_parameters; stripping",
                        resolved_model, optional_param,
                    )
                    kwargs.pop(optional_param, None)
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
        # An HTTP-200 that carried a provider body-error (OpenRouter passes
        # 429/5xx through the body) reaches here only when a same-model reroute
        # was unavailable or also errored. Surface it as a typed marker so the
        # caller classifies it as a real rate_limit/provider_transient instead of
        # a blank finish_reason=null "incomplete response".
        _body_err = self._provider_body_error(resp_dict)
        if _body_err:
            usage["provider_error"] = {
                "code": _body_err.get("code"),
                "message": str(_body_err.get("message") or "")[:300],
                "kind": "rate_limit" if self._is_transient_body_error(_body_err) and str(_body_err.get("code")) == "429"
                else ("provider_transient" if self._is_transient_body_error(_body_err) else "provider_error"),
            }
        choices = resp_dict.get("choices") or [{}]
        msg = dict((choices[0] if choices else {}).get("message") or {})
        if resp_dict.get("id") and "response_id" not in msg:
            msg["response_id"] = resp_dict["id"]

        # OpenAI SDK model_dump() adds nullable fields that strict OpenAI-compatible
        # providers reject as extra inputs when the message re-enters conversation history.
        for _sdk_field in ("refusal", "annotations", "audio", "function_call"):
            if msg.get(_sdk_field) is None:
                msg.pop(_sdk_field, None)
        annotations = msg.get("annotations") if isinstance(msg.get("annotations"), list) else []
        web_sources: List[Dict[str, str]] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            citation = annotation.get("url_citation") if isinstance(annotation.get("url_citation"), dict) else annotation
            url = str(citation.get("url") or "").strip() if isinstance(citation, dict) else ""
            if not url:
                continue
            web_sources.append({
                "url": url[:500],
                "title": str(citation.get("title") or "")[:300] if isinstance(citation, dict) else "",
                "content": str(citation.get("content") or citation.get("snippet") or "")[:1000] if isinstance(citation, dict) else "",
            })
        if web_sources:
            usage["web_search_sources"] = web_sources[:20]
        # Provider response annotations are transport metadata, not valid chat
        # input fields for the next round. Persist harvested citations in usage.
        msg.pop("annotations", None)
        if isinstance(usage.get("server_tool_use"), dict):
            usage["server_tool_use"] = dict(usage["server_tool_use"])
        # Provider-private reasoning text on the OpenAI-compatible direct lanes
        # (GLM / Z.AI / cloud.ru, legacy vLLM expose a top-level ``reasoning_content``).
        # Unlike ``reasoning``/``reasoning_details`` (kept for same-family continuity
        # and scrubbed only on a cross-family switch), strict vLLM/SGLang servers reject
        # their OWN echoed ``reasoning_content`` with a 400 ``Extra inputs are not
        # permitted`` on the very next same-model turn. Drop it here so it never enters
        # the canonical transcript; the outbound scrubber is the second layer.
        msg.pop("reasoning_content", None)

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
        # v6.61.1 (Q7 disclosure): a learned-ceiling clamp recorded at payload build
        # (_build_remote_kwargs → _clamp_effort_for_model) rides THIS call's usage —
        # covers both the OpenRouter and the OpenAI-compatible direct lanes.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note

        return msg, usage

    @staticmethod
    def extract_display_reasoning(msg: Dict[str, Any]) -> str:
        """Provider-agnostic, SHAPE-based reader for human-readable reasoning to NARRATE in an
        otherwise-empty tool-round bubble. Reads only the readable forms a provider may already
        leave on the normalized message — flat ``reasoning`` (OpenRouter / some OpenAI-compatible),
        structured ``reasoning_details`` of readable types, or ``content`` thinking/thought blocks
        (Anthropic ``thinking`` / Gemini ``part.thought``) — and SKIPS opaque/encrypted payloads
        (``reasoning.encrypted``, ``redacted_thinking``, signature/data-only blocks), which carry no
        display text and must round-trip byte-for-byte. DISPLAY-ONLY: the caller keeps the result in
        a local variable and never appends it to the transcript nor sends it to a provider — the raw
        fields it reads are already on the message and handled by the outbound scrubbers."""
        if not isinstance(msg, dict):
            return ""
        parts: List[str] = []

        flat = msg.get("reasoning")
        if isinstance(flat, str) and flat.strip():
            parts.append(flat.strip())

        details = msg.get("reasoning_details")
        if isinstance(details, list):
            for d in details:
                if not isinstance(d, dict):
                    continue
                if str(d.get("type") or "") in ("reasoning.text", "reasoning.summary"):
                    txt = d.get("text") or d.get("summary")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                # reasoning.encrypted / signature / data-only payloads are opaque -> skipped.

        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type") or "")
                if btype == "thinking":
                    txt = block.get("thinking")
                elif btype == "reasoning":
                    txt = block.get("text") or block.get("reasoning")
                elif block.get("thought") is True:  # Gemini part.thought == true
                    txt = block.get("text")
                else:
                    continue  # text / tool_use / redacted_thinking / encrypted -> not display text
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())

        # De-dup across the whole set (order-preserving): a provider often carries the SAME
        # readable rollup in both flat ``reasoning`` and a ``reasoning.summary`` detail (verified
        # against live gpt-5.5), so a consecutive-only check would still double it.
        deduped: List[str] = []
        seen: Set[str] = set()
        for p in parts:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return "\n".join(deduped).strip()

    def _create_chat_completion_with_retries(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        usage_model = str(target.get("usage_model") or target.get("resolved_model") or "")
        try:
            resp = create_fn(**kwargs)
        except Exception as exc:
            retry_kwargs = self._retry_without_optional_sampling(kwargs, usage_model, exc)
            if retry_kwargs is not None:
                try:
                    return create_fn(**retry_kwargs)
                except Exception as retry_exc:
                    stripped_kwargs = self._openrouter_signature_retry_kwargs(target, retry_kwargs, retry_exc)
                    if stripped_kwargs is None:
                        raise
                    return create_fn(**stripped_kwargs)
            stripped_kwargs = self._openrouter_signature_retry_kwargs(target, kwargs, exc)
            if stripped_kwargs is None:
                raise
            return create_fn(**stripped_kwargs)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                return create_fn(**reroute_kwargs)
            except Exception:
                return resp
        return resp

    async def _create_chat_completion_with_retries_async(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        usage_model = str(target.get("usage_model") or target.get("resolved_model") or "")
        try:
            resp = await create_fn(**kwargs)
        except Exception as exc:
            retry_kwargs = self._retry_without_optional_sampling(kwargs, usage_model, exc)
            if retry_kwargs is not None:
                try:
                    return await create_fn(**retry_kwargs)
                except Exception as retry_exc:
                    stripped_kwargs = self._openrouter_signature_retry_kwargs(target, retry_kwargs, retry_exc)
                    if stripped_kwargs is None:
                        raise
                    return await create_fn(**stripped_kwargs)
            stripped_kwargs = self._openrouter_signature_retry_kwargs(target, kwargs, exc)
            if stripped_kwargs is None:
                raise
            return await create_fn(**stripped_kwargs)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                return await create_fn(**reroute_kwargs)
            except Exception:
                return resp
        return resp

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
        timeout: Optional[float] = None,
        allow_server_web_search: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send remote chat; no_proxy uses a one-shot client and skips OS proxy lookup."""
        if target.get("provider") == "anthropic":
            return self._chat_anthropic(
                target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                no_proxy=no_proxy,
                timeout=timeout,
            )

        if target.get("provider") == "gigachat":
            return self._chat_gigachat(
                target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                no_proxy=no_proxy,
                timeout=timeout,
            )

        if no_proxy:
            _oa_client, _http_client = self._make_no_proxy_client(target, timeout=timeout)
            try:
                kwargs = self._build_remote_kwargs(
                    target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
                    skip_capability_fetch=True,
                    allow_server_web_search=allow_server_web_search,
                    response_format=response_format,
                )
                prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
                    kwargs.get("messages"),
                    kwargs.get("tools"),
                )
                resp = self._create_chat_completion_with_retries(
                    _oa_client.chat.completions.create,
                    kwargs,
                    target,
                )
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
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
            allow_server_web_search=allow_server_web_search,
            response_format=response_format,
        )
        if timeout and timeout > 0:
            # Cached clients are built without a timeout; honor the caller's
            # per-request timeout instead of silently using the SDK default.
            kwargs["timeout"] = float(timeout)
        prompt_cache_ttl = self._prompt_cache_ttl_from_payload(
            kwargs.get("messages"),
            kwargs.get("tools"),
        )
        resp = self._create_chat_completion_with_retries(
            client.chat.completions.create,
            kwargs,
            target,
        )
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
        reasoning_effort: str = "medium",
        timeout: float = 90.0,
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
            no_proxy=True,
            timeout=timeout,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "google/gemini-3.5-flash")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "google/gemini-3.5-flash")
        heavy = os.environ.get("OUROBOROS_MODEL_HEAVY", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if heavy and heavy != main:
            models.append(heavy)
        if light and light != main and light != heavy:
            models.append(light)
        return models


def openrouter_web_search_server_tool(
    *,
    api_key: str,
    model: str,
    query: str,
    search_context_size: str,
) -> Any:
    """Run OpenRouter's provider-owned web_search server tool."""

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools=[{
            "type": "openrouter:web_search",
            "parameters": {
                "search_context_size": search_context_size,
                "max_total_results": 10,
            },
        }],
    )


def anthropic_web_search_server_tool(
    *,
    api_key: str,
    model: str,
    query: str,
) -> Any:
    """Run Anthropic's provider-owned web_search server tool."""

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    return client.messages.create(
        model=model,
        max_tokens=2048,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": query}],
    )
