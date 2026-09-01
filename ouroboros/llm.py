"""LLM client for OpenRouter, direct providers, and optional local inference."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.anthropic_native_custody import (
    anthropic_replay_scoped,
    custody_private_key,
    is_replayed_native_content,
    mark_replayed_receipts_consumed,
    native_content_for_replay,
    retain_native_assistant_content,
    scrub_native_custody,
)
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from ouroboros.provider_models import OPENROUTER_DEFAULTS, PROVIDER_PREFIXES, normalize_anthropic_model_id, normalize_model_identity, resolve_minimax_base_url
from ouroboros.request_wire_recovery import (
    finalize_wire_response,
    note_provider_metadata_drop_fields,
    note_wire_send_failed,
    note_wire_send_succeeded,
    plan_next_wire_retry,
    prepare_wire_payload_for_send,
    request_wire_scoped,
)
from ouroboros.usage_accounting import (
    AttemptRequest,
    PhysicalAttemptCapture,
    PhysicalAttemptPreconditionFailed,
    PhysicalAttemptPreparationFailed,
    UsageAccountingError,
    UsageScope,
    capture_attempt_ids,
    current_physical_attempt_context,
    current_physical_attempt_predicate,
    current_usage_scope,
    execute_physical_attempt,
    execute_physical_attempt_async,
    last_physical_attempt_capture,
    usage_scope,
)
from ouroboros.transport_custody import is_loopback_base_url
from ouroboros.utils import in_worker_process, sanitize_tool_result_for_log

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = OPENROUTER_DEFAULTS["light"]
_FALSE_LIKE_ENV_VALUES = {"", "0", "false", "no", "off"}
# Provider-valid Anthropic ephemeral-cache tiers.
_VALID_CACHE_TTLS = frozenset({"5m", "1h"})

# Only explicit wire tiers have a knowable horizon; bare "default" does not.
_CACHE_TTL_SECONDS = {"5m": 300, "1h": 3600}

from ouroboros.context_budget import (
    CONTEXT_OVERFLOW_CODES,
    context_overflow_message,
)
from ouroboros.tool_call_markup import (
    parse_tool_calls_from_content,
    strip_reasoning_wrappers,
)

# Response-only labels are diagnostic facts, not canonical assistant fields. Keep
# provider-supplied values bounded and printable before they enter usage custody.
_RESPONSE_METADATA_LABEL_MAX_CHARS = 160


def _bounded_response_metadata_label(value: Any) -> Optional[str]:
    """Return a small printable response label, or omit an unsafe/unknown value."""
    if not isinstance(value, str):
        return None
    label = value.strip()
    if not label or len(label) > _RESPONSE_METADATA_LABEL_MAX_CHARS:
        return None
    if not label.isprintable():
        return None
    if sanitize_tool_result_for_log(label) != label:
        return None
    return label


def _structured_error_values(payload: Any) -> Set[str]:
    if not isinstance(payload, dict):
        return set()
    nodes = [payload]
    if isinstance(payload.get("error"), dict):
        nodes.append(payload["error"])
    return {
        str(node.get(key) or "").strip().lower()
        for node in nodes
        for key in ("code", "type")
        if str(node.get(key) or "").strip()
    }


def _is_structured_context_overflow_exception(exc: BaseException) -> bool:
    """Read only facts attached to this exception; never a stale ContextVar."""
    values = {
        str(getattr(exc, key, "") or "").strip().lower()
        for key in ("code", "type")
        if str(getattr(exc, key, "") or "").strip()
    }
    values.update(_structured_error_values(getattr(exc, "body", None)))
    capture = getattr(exc, "physical_attempt_capture", None)
    if capture is not None:
        values.update({
            str(getattr(capture, key, "") or "").strip().lower()
            for key in ("provider_code", "provider_error_type")
            if str(getattr(capture, key, "") or "").strip()
        })
    return bool(values & CONTEXT_OVERFLOW_CODES)


def _is_structured_context_overflow_body(error: Any) -> bool:
    return bool(_structured_error_values(error) & CONTEXT_OVERFLOW_CODES)


def cache_ttl_seconds(applied_ttl: Any) -> Optional[int]:
    """Return seconds only for an explicit TTL carried by the candidate."""
    return _CACHE_TTL_SECONDS.get(str(applied_ttl or "").strip().lower())


def supports_message_cache_control(model: str) -> bool:
    """Whether the OpenRouter family honors message cache breakpoints."""
    m = str(model or "").strip().lstrip("~")
    return m.startswith("anthropic/") or m.startswith("google/gemini-")


def _route_normalizes_cache_breakpoints(target: Dict[str, Any]) -> bool:
    """Whether the send-time finalizer may normalize cache breakpoints."""
    if str(target.get("provider") or "") == "anthropic":
        return True
    model = str(target.get("resolved_model") or "").strip().lstrip("~")
    return bool(
        target.get("supports_openrouter_extensions")
        and supports_message_cache_control(model)
        and model.startswith("anthropic/")
    )


def _reasoning_signature_portable_across_or_providers(model: str) -> bool:
    """Whether replay signatures are verified portable across same-model providers."""
    m = str(model or "").strip().lstrip("~")
    return (
        m.startswith("anthropic/")
        or m.startswith("google/gemini-")
        or m.startswith("openai/")
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
_OPTIONAL_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
# Provider-rejected optional intent may be removed by the one-shot retry ladder.
_OPTIONAL_DROPPABLE_PARAMS = _OPTIONAL_SAMPLING_PARAMS + (
    "response_format", "reasoning_effort", "output_config", "thinking",
)
# Shared by the classifier and floor predicate; bare "required" is too broad.
_MANDATORY_VALUE_MARKERS = ("mandatory", "cannot be disabled", "must be enabled")


class LocalContextTooLargeError(RuntimeError):
    """Raised when a local model cannot fit context without silent truncation."""


# Lives beside its proxy constant; the historical private name stays importable.
from ouroboros.context_budget import estimate_message_chars as _estimate_message_chars


def _applied_payload_cache_ttl(payload: Dict[str, Any]) -> Optional[str]:
    """Strongest cache TTL carried by THIS exact candidate payload.

    Same reporting rule as the send-time finalizer's return value
    (``_normalize_payload_cache_ttl``: 1h > 5m > bare markers = "default";
    None when the payload carries no markers). Read per candidate rather than
    plumbed from the finalizer because the retry ladder can strip markers
    (``_retry_without_prompt_cache_parameter``) after the finalizer ran — the
    reservation must price the payload actually being sent, not the original.
    """
    breakpoints = LLMClient._payload_cache_breakpoints(payload)
    ttls = {
        str((holder.get("cache_control") or {}).get("ttl") or "").strip().lower()
        for holder in breakpoints
    }
    if "1h" in ttls:
        return "1h"
    if "5m" in ttls:
        return "5m"
    return "default" if breakpoints else None


def _attempt_request(
    target: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    source: Optional[str] = None,
) -> AttemptRequest:
    """Build secret-free facts for one final inspectable candidate."""
    prompt_payload = {
        key: value
        for key, value in payload.items()
        if key not in {
            "model", "max_tokens", "max_completion_tokens", "temperature",
            "top_p", "top_k", "timeout", "stream",
        }
    }
    try:
        prompt_chars = len(json.dumps(prompt_payload, ensure_ascii=False, default=str))
    except Exception:
        prompt_chars = len(str(prompt_payload or ""))
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload

    bounded_tokens = bounded_prompt_tokens_for_payload(prompt_payload, prompt_chars)
    request_source = source
    if request_source is None:
        bound_scope = current_usage_scope()
        request_source = (
            str(bound_scope.source)
            if bound_scope is not None and bound_scope.source
            else "llm.chat"
        )
    raw = _canonical_candidate_bytes(payload)
    context = _canonical_candidate_bytes({
        key: payload[key] for key in ("system", "messages", "tools", "functions") if key in payload
    })
    return AttemptRequest(
        model=str(target.get("usage_model") or target.get("resolved_model") or payload.get("model") or ""),
        provider=str(target.get("provider") or "unknown"),
        prompt_tokens_estimate=max(0, prompt_chars // 4),
        max_completion_tokens=int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 0),
        source=str(request_source or ""),
        prompt_cache_ttl=_applied_payload_cache_ttl(payload) or "",
        candidate_raw_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_raw_size_bytes=len(raw),
        candidate_context_sha256=hashlib.sha256(context).hexdigest(),
        candidate_context_size_bytes=len(context),
        candidate_measurement_kind="canonical_json_v1",
        physical_context=current_physical_attempt_context(),
        route_is_loopback=is_loopback_base_url(target.get("base_url")),
        prompt_tokens_bounded_estimate=bounded_tokens,
    )


def _canonical_candidate_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def _physical_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the send copy with capsule metadata removed only from context turns."""
    candidate = copy.deepcopy(payload)

    def _strip(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("_context_capsule", None)
            for child in value.values():
                _strip(child)
        elif isinstance(value, list):
            for child in value:
                _strip(child)

    for key in ("system", "messages"):
        _strip(candidate.get(key))
    return candidate


def _candidate_before_dispatch(candidate: Dict[str, Any], request: AttemptRequest):
    """Close over one final candidate without putting it in accounting rows."""
    predicate = current_physical_attempt_predicate()

    def _persist(reservation):
        fresh = _attempt_request(
            {"provider": request.provider, "usage_model": request.model}, candidate,
            source=request.source,
        )
        identity = (
            fresh.candidate_raw_sha256, fresh.candidate_raw_size_bytes,
            fresh.candidate_context_sha256, fresh.candidate_context_size_bytes,
        )
        expected = (
            request.candidate_raw_sha256, request.candidate_raw_size_bytes,
            request.candidate_context_sha256, request.candidate_context_size_bytes,
        )
        if identity != expected:
            raise PhysicalAttemptPreparationFailed(
                "physical candidate changed before dispatch", attempt_id=reservation.attempt_id,
            )
        from ouroboros.observability import persist_physical_candidate

        scope = current_usage_scope()
        persisted = persist_physical_candidate(
            reservation.drive_root,
            task_id=str(scope.task_id if scope is not None else request.task_id),
            attempt_id=reservation.attempt_id,
            candidate=candidate,
            candidate_facts={
                "candidate_raw_sha256": request.candidate_raw_sha256,
                "candidate_raw_size_bytes": request.candidate_raw_size_bytes,
                "candidate_context_sha256": request.candidate_context_sha256,
                "candidate_context_size_bytes": request.candidate_context_size_bytes,
                "candidate_measurement_kind": request.candidate_measurement_kind,
                "physical_context": (
                    dict(vars(request.physical_context)) if request.physical_context is not None else None
                ),
            },
        )
        if predicate is not None:
            try:
                accepted = predicate(request)
            except BaseException as exc:
                # Persistence already succeeded. Preserve the only durable link
                # even when the host predicate itself raises before returning.
                try:
                    exc.candidate_manifest_ref = persisted["manifest_ref"]
                except Exception:
                    pass
                raise
            if accepted is False:
                failure = PhysicalAttemptPreconditionFailed(
                    "physical candidate precondition rejected dispatch",
                    attempt_id=reservation.attempt_id,
                )
                failure.candidate_manifest_ref = persisted["manifest_ref"]
                raise failure
        return persisted["manifest_ref"]

    return _persist


def _execute_candidate(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    """Keep existing two-argument injected executors usable."""
    if "before_dispatch" not in inspect.signature(execute_physical_attempt).parameters:
        return execute_physical_attempt(request, send)
    return execute_physical_attempt(request, send, before_dispatch=before_dispatch)


async def _execute_candidate_async(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    if "before_dispatch" not in inspect.signature(execute_physical_attempt_async).parameters:
        return await execute_physical_attempt_async(request, send)
    return await execute_physical_attempt_async(request, send, before_dispatch=before_dispatch)


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
        allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    from ouroboros.request_wire_recovery import merge_request_wire_usage

    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost") is not None:
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])
        if usage.get("cost_final") is False or usage.get("cost_estimated"):
            total["cost_final"] = False
    else:
        total["cost_final"] = False
    merge_request_wire_usage(total, usage)


def fetch_openrouter_pricing(*, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Fetch OpenRouter pricing as model_id -> per-1M prices.

    Tuples are ``(input, cached_read, cache_write, output)``. Missing cache
    prices remain ``None`` instead of inheriting a synthetic coefficient.
    """
    import logging
    from ouroboros.pricing import PricingSchedule
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=max(0.1, min(5.0, float(timeout_sec))))
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        pricing_dict = {}
        for model in models:
            model_id = str(model.get("id") or "").strip()

            pricing = model.get("pricing", {})
            if not pricing or pricing.get("prompt") is None or pricing.get("completion") is None:
                continue

            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str is not None else None
            raw_cache_write_str = pricing.get("input_cache_write")
            raw_cache_write = float(raw_cache_write_str) if raw_cache_write_str is not None else None
            if raw_prompt < 0 or raw_completion < 0:
                continue
            if raw_cached is not None and raw_cached < 0:
                raw_cached = None
            if raw_cache_write is not None and raw_cache_write < 0:
                raw_cache_write = None

            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            cached_price = round(raw_cached * 1_000_000, 4) if raw_cached is not None else None
            cache_write_price = (
                round(raw_cache_write * 1_000_000, 4)
                if raw_cache_write is not None else None
            )

            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            row = (prompt_price, cached_price, cache_write_price, completion_price)

            tiers = []
            raw_overrides = pricing.get("overrides") or []
            if isinstance(raw_overrides, list):
                for override in raw_overrides:
                    if not isinstance(override, dict):
                        continue
                    try:
                        min_prompt_tokens = int(override.get("min_prompt_tokens") or 0)
                        if min_prompt_tokens <= 0:
                            continue
                        tier_raw_prompt = float(override.get("prompt", raw_prompt))
                        tier_raw_completion = float(override.get("completion", raw_completion))
                        tier_prompt = round(tier_raw_prompt * 1_000_000, 4)
                        tier_completion = round(tier_raw_completion * 1_000_000, 4)
                        override_cached = override.get("input_cache_read")
                        tier_cached = (
                            round(float(override_cached) * 1_000_000, 4)
                            if override_cached is not None else None
                        )
                        override_write = override.get("input_cache_write")
                        if override_write is not None:
                            tier_write = round(float(override_write) * 1_000_000, 4)
                        else:
                            tier_write = None
                        if tier_prompt > 1000 or tier_completion > 1000:
                            continue
                        tier_row = (tier_prompt, tier_cached, tier_write, tier_completion)
                        tiers.append((min_prompt_tokens, tier_row))
                    except (TypeError, ValueError):
                        log.warning("Skipping malformed pricing override for %s", model_id)
            if tiers:
                row = PricingSchedule(row, tuple(tiers))
            pricing_dict[model_id] = row
            normalized_model_id = normalize_model_identity(model_id)
            if normalized_model_id != model_id:
                pricing_dict[normalized_model_id] = row

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def fetch_cloudru_pricing(*, timeout_sec: float = 5.0) -> Dict[str, Tuple[Optional[float], ...]]:
    """Fetch cloud.ru Foundation Models pricing as ``cloudru/<id>`` -> per-1M USD.

    cloud.ru's ``GET /v1/models`` returns per-model ``metadata`` with token costs
    (``prompt_tokens_cost``, ``generated_tokens_cost``, ``cache_read_tokens_cost``,
    ``cache_write_tokens_cost``) in RUB per 1M tokens — i.e. the real resale price
    the owner pays. We convert to USD via ``OUROBOROS_RUB_USD_RATE`` so the catalog
    is the SSOT for ALL cloud.ru models (no hardcoded per-model table). Models with
    ``is_billable=false`` is an exact free row; missing billability or an absent
    explicit ``OUROBOROS_RUB_USD_RATE`` stays unknown. Returns {} when the catalog
    cannot be queried. Tuples are ``(input, cached_read, cache_write, output)``."""
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
        rate = float(os.environ.get("OUROBOROS_RUB_USD_RATE", ""))
    except (TypeError, ValueError):
        return {}
    if rate <= 0:
        return {}

    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=max(0.1, min(5.0, float(timeout_sec))),
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

        pricing_dict: Dict[str, Tuple[Optional[float], ...]] = {}
        for model in models:
            model_id = str(model.get("id") or "").strip()
            meta = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
            if not model_id or not meta or meta.get("is_billable") is None:
                continue
            if meta.get("is_billable") is False:
                pricing_dict[normalize_model_identity(f"cloudru::{model_id}")] = (0.0, 0.0, 0.0, 0.0)
                continue
            prompt_price = _rub_per_1m_to_usd(meta.get("prompt_tokens_cost"))
            output_price = _rub_per_1m_to_usd(meta.get("generated_tokens_cost"))
            if prompt_price is None or output_price is None:
                continue
            cached_price = _rub_per_1m_to_usd(meta.get("cache_read_tokens_cost"))
            cache_write_price = _rub_per_1m_to_usd(meta.get("cache_write_tokens_cost"))
            row = (
                prompt_price,
                cached_price,
                cache_write_price,
                output_price,
            )
            pricing_dict[normalize_model_identity(f"cloudru::{model_id}")] = row

        log.info(f"Fetched pricing for {len(pricing_dict)} models from cloud.ru")
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
    # Splits provider OUTAGE from a route with no metadata, so Capability Evidence
    # can mark STATUS_FAILED (transient) vs STATUS_UNPROBEABLE (v6.33.0 P4).
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
            # 5s, not 15s: this fetch sits on the synchronous capability-probe path
            # behind the max-context-mode gate (settings save / max toggle); a slow
            # probe must fail-closed quickly, never hang the save (v6.33.0 WS4).
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
        """Legacy model-global diagnostic; production uses request-wire recovery."""
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
                "not permitted",
                # VALUE-rejection families (v6.73.2). Mandatory-enable family —
                # the parameter is supported but its DISABLED/bottom value is
                # forbidden (e.g. Gemini "Reasoning is mandatory for this
                # endpoint and cannot be disabled"); routed to the effort-FLOOR
                # branch by _mandatory_value_rejection (which consumes the SAME
                # _MANDATORY_VALUE_MARKERS constant), never to the drop path
                # for effort carriers.
                *_MANDATORY_VALUE_MARKERS,
                # Range/value family — the VALUE is out of the accepted range
                # (e.g. "temperature must be between 0 and 2"). These take the
                # existing drop path: the param is an optional hint and removing
                # it is the correct degradation.
                "must be between",
                "out of range",
                "invalid value",
            )
        )

    @staticmethod
    def _mandatory_value_rejection(exc: BaseException) -> bool:
        """True when a provider rejected a parameter VALUE as 'must stay enabled'
        (reasoning cannot be turned off) rather than the parameter being
        unsupported. Only ever consulted AFTER _parameter_rejection_error matched
        (which already required a droppable-param name in the text), so a bare
        marker here cannot fire on unrelated errors. This is the gate that sends
        a bottom-tier effort rejection to the FLOOR branch (raise + learn floor)
        instead of the drop path — value-forbidden and capability-absent need
        OPPOSITE remedies."""
        text = str(exc or "").lower()
        if not text:
            return False
        return any(m in text for m in _MANDATORY_VALUE_MARKERS)

    # Durable twin of _REJECTED_PARAMS_CACHE (v6.69.0): learned rejections survive
    # process/restart boundaries via capability_evidence (same design as the
    # effort-ceiling cache below — normalized-model-identity key, fail-open, and
    # entries expire so a provider re-enabling a parameter heals itself). The
    # process cache re-syncs from the durable store hourly so the 14-day expiry
    # also heals LONG-RUNNING processes, not only restarts.
    _REJECTED_PARAMS_LOADED: Dict[str, float] = {}
    _REJECTED_PARAMS_RELOAD_SEC = 3600.0

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
        try:
            from ouroboros.capability_evidence import record_rejected_params
            from ouroboros.config import DATA_DIR
            durable_key = normalize_model_identity(model_id) or str(model_id)
            record_rejected_params(DATA_DIR, durable_key, params)
        except Exception:
            pass

    @classmethod
    def _known_rejected_params(cls, model_id: str) -> Set[str]:
        if not model_id:
            return set()
        out: Set[str] = set()
        durable_key = normalize_model_identity(model_id) or str(model_id)
        now = time.monotonic()
        loaded_at = cls._REJECTED_PARAMS_LOADED.get(durable_key)
        if durable_key and (
            loaded_at is None or now - loaded_at >= cls._REJECTED_PARAMS_RELOAD_SEC
        ):
            cls._REJECTED_PARAMS_LOADED[durable_key] = now
            try:
                from ouroboros.capability_evidence import get_rejected_params
                from ouroboros.config import DATA_DIR
                cls._REJECTED_PARAMS_CACHE[durable_key] = set(get_rejected_params(DATA_DIR, durable_key))
            except Exception:
                pass
        for key in {model_id, normalize_model_identity(model_id)}:
            out.update(cls._REJECTED_PARAMS_CACHE.get(key, set()))
        return out

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
    _EFFORT_FLOOR_CACHE: Dict[str, str] = {}
    _EFFORT_FLOOR_LOADED: Dict[str, float] = {}
    _EFFORT_FLOOR_RELOAD_SEC = 3600.0

    @classmethod
    def _effort_floor_for(cls, model_id: str) -> str:
        key = normalize_model_identity(model_id) or str(model_id or "")
        if not key:
            return ""
        now = time.monotonic()
        loaded_at = cls._EFFORT_FLOOR_LOADED.get(key)
        if loaded_at is None or now - loaded_at >= cls._EFFORT_FLOOR_RELOAD_SEC:
            cls._EFFORT_FLOOR_LOADED[key] = now
            try:
                from ouroboros.capability_evidence import get_effort_floor
                from ouroboros.config import DATA_DIR
                # Replace (not union): the durable reader applies the 14-day
                # expiry, so replacing lets an expired floor actually evict.
                cls._EFFORT_FLOOR_CACHE[key] = get_effort_floor(DATA_DIR, key)
            except Exception:
                pass
        return cls._EFFORT_FLOOR_CACHE.get(key, "")

    @classmethod
    def _record_effort_floor(cls, model_id: str, floor: str) -> None:
        """A provider rejected a bottom-tier effort as 'reasoning is mandatory' →
        learn the route's minimum. In-process + durable (14-day expiry there), so
        subsequent calls clamp UP immediately. Higher floor wins in-process too."""
        from ouroboros.config import effort_rank
        key = normalize_model_identity(model_id) or str(model_id or "")
        value = str(floor or "").strip().lower()
        if not key or not value:
            return
        prev = cls._EFFORT_FLOOR_CACHE.get(key, "")
        if not prev or effort_rank(value) > effort_rank(prev):
            cls._EFFORT_FLOOR_CACHE[key] = value
        cls._EFFORT_FLOOR_LOADED[key] = time.monotonic()
        try:
            from ouroboros.capability_evidence import record_effort_floor
            from ouroboros.config import DATA_DIR
            record_effort_floor(DATA_DIR, key, value)
        except Exception:
            pass

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

    @classmethod
    def clamp_effort_for_route(cls, model_id: str, effort: str) -> str:
        """Resolve the legacy diagnostic model-global effort band."""
        ceiling = cls._effort_ceiling_for(model_id)
        floor = cls._effort_floor_for(model_id)
        if not ceiling and not floor:
            return effort
        from ouroboros.config import clamp_effort_to, effort_rank
        applied = clamp_effort_to(effort, ceiling) if ceiling else effort
        if floor and 0 <= effort_rank(applied) < effort_rank(floor):
            applied = floor
        return applied

    def _clamp_effort_for_model(self, model_id: str, effort: str) -> str:
        """Legacy clamp plus diagnostic disclosure; production dispatch bypasses it."""
        if not hasattr(self, "_effort_clamp_tls"):
            self._effort_clamp_tls = threading.local()
        self._effort_clamp_tls.pending = None
        from ouroboros.config import effort_rank
        applied = self.clamp_effort_for_route(model_id, effort)
        if applied != effort:
            self._effort_clamp_tls.pending = {
                "requested": effort,
                "applied": applied,
                "reason": (
                    "learned_floor"
                    if effort_rank(applied) > effort_rank(effort)
                    else "learned_ceiling"
                ),
                "model": str(model_id or ""),
            }
        return applied

    def _pop_effort_clamp_disclosure(self) -> Optional[Dict[str, Any]]:
        """The pending clamp record for THIS thread's in-flight call, if any."""
        tls = getattr(self, "_effort_clamp_tls", None)
        pending = getattr(tls, "pending", None) if tls is not None else None
        if tls is not None:
            tls.pending = None
        return pending if isinstance(pending, dict) else None

    @classmethod
    def _record_effort_ceiling(cls, model_id: str, current_effort: str) -> None:
        """Record a legacy model-global ceiling for diagnostics."""
        from ouroboros.config import effort_one_step_down, effort_rank
        key = normalize_model_identity(model_id) or str(model_id or "")
        eff = str(current_effort or "").strip().lower()
        if not key or not eff:
            return
        ceiling = effort_one_step_down(eff)
        if effort_rank(ceiling) < effort_rank("low"):
            return
        prev = cls._EFFORT_CEILING_CACHE.get(key)
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

    @staticmethod
    def _set_payload_effort(payload: Dict[str, Any], effort: str) -> None:
        """Write effort into each carrier already present in the payload."""
        if "reasoning_effort" in payload:
            payload["reasoning_effort"] = effort
        oc = payload.get("output_config")
        if isinstance(oc, dict) and "effort" in oc:
            oc["effort"] = effort
        eb = payload.get("extra_body")
        if isinstance(eb, dict) and isinstance(eb.get("reasoning"), dict):
            eb["reasoning"]["effort"] = effort

    def _retry_without_optional_sampling(
        self,
        payload: Dict[str, Any],
        model_id: str,
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        cls = type(self)
        if _is_structured_context_overflow_exception(exc):
            return None
        if not cls._parameter_rejection_error(exc):
            return None
        _err_text = str(exc or "").lower()
        _effort_implicated = any(
            k in _err_text for k in ("reasoning_effort", "output_config", "thinking", "reasoning", "effort")
        )
        if cls._mandatory_value_rejection(exc):
            requested = cls._payload_effort(payload)
            if not _effort_implicated or requested not in ("none", "minimal"):
                return None
            cls._record_effort_floor(model_id, "low")
            applied = self._clamp_effort_for_model(model_id, requested)
            if applied == requested:
                return None
            retry_payload = copy.deepcopy(payload)
            cls._set_payload_effort(retry_payload, applied)
            log.warning(
                "Retrying %s with reasoning effort raised to learned floor %r "
                "(provider requires reasoning enabled)",
                model_id or "(unknown model)", applied,
            )
            return retry_payload
        present = {param for param in _OPTIONAL_DROPPABLE_PARAMS if param in payload}
        _err_compact = _err_text.replace(".", "_")
        _named = {param for param in present if param in _err_text or param in _err_compact}
        _eb = payload.get("extra_body")
        _nested_reasoning = isinstance(_eb, dict) and isinstance(_eb.get("reasoning"), dict)
        if _nested_reasoning and _effort_implicated:
            _named.add(cls._NESTED_REASONING_PARAM)
            present.add(cls._NESTED_REASONING_PARAM)
        if _named:
            if _named & {"thinking", "output_config"}:
                _named |= {"thinking", "output_config"} & present
            present = _named
        if not present:
            return None
        if (
            present & {"reasoning_effort", "output_config", "thinking", cls._NESTED_REASONING_PARAM}
            and _effort_implicated
        ):
            cls._record_effort_ceiling(model_id, cls._payload_effort(payload))
        cls._remember_rejected_params(model_id, present)
        retry_payload = copy.deepcopy(payload)
        for param in present:
            if param == cls._NESTED_REASONING_PARAM:
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
    def _retry_without_prompt_cache_parameter(
        payload: Dict[str, Any],
        target: Dict[str, Any],
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        """Remove only an explicitly rejected cache control or affinity once."""
        if _is_structured_context_overflow_exception(exc):
            return None
        provider = str(target.get("provider") or "").strip().lower()
        extra_body = payload.get("extra_body")
        param = ""
        if provider == "openai" and "prompt_cache_key" in payload:
            param = "prompt_cache_key"
        elif (
            bool(target.get("supports_openrouter_extensions"))
            and isinstance(extra_body, dict)
            and "session_id" in extra_body
        ):
            param = "session_id"
        elif (
            provider == "openai-compatible"
            and isinstance(extra_body, dict)
            and "cache" in extra_body
        ):
            param = "cache"
        if not param:
            return None

        text = str(exc or "").lower()
        if param not in text:
            return None
        if not any(
            marker in text
            for marker in (
                "unsupported",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "unexpected keyword",
                "unexpected field",
                "invalid parameter",
                "not permitted",
                "extra inputs",
                "additional properties",
                "no endpoints found",
                "requested parameter",
            )
        ):
            return None

        retry_payload = copy.deepcopy(payload)
        if param == "prompt_cache_key":
            retry_payload.pop(param, None)
        else:
            retry_extra = retry_payload.get("extra_body")
            if isinstance(retry_extra, dict):
                retry_extra.pop(param, None)
            if not retry_extra:
                retry_payload.pop("extra_body", None)
        log.warning(
            "Retrying %s once without unsupported cache parameter %s",
            str(target.get("usage_model") or target.get("resolved_model") or "(unknown model)"),
            param,
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
        if provider == "minimax":
            return f"minimax/{resolved_model}"
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

    @classmethod
    def _copy_messages_with_cache_policy(
        cls,
        messages: List[Dict[str, Any]],
        *,
        allow_message_cache_control: bool,
        flatten_tool_content_blocks: bool,
        allow_cache_ttl: bool = False,
    ) -> List[Dict[str, Any]]:
        cleaned = scrub_native_custody(messages)
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
                        # Strict providers reject cache markers on empty text.
                        empty_text = (
                            block.get("type") == "text"
                            and not str(block.get("text") or "").strip()
                        )
                        if (allow_message_cache_control
                                and isinstance(block.get("cache_control"), dict)
                                and not empty_text):
                            # Keep TTL only where the route documents it.
                            ttl = str(block["cache_control"].get("ttl") or "")
                            block["cache_control"] = (
                                {"type": "ephemeral", "ttl": ttl}
                                if allow_cache_ttl and ttl in _VALID_CACHE_TTLS
                                else {"type": "ephemeral"}
                            )
                        else:
                            block.pop("cache_control", None)
                        # Known host metadata never leaves the send copy.
                        for key in ("_caption", "_source_path", "_context_capsule"):
                            block.pop(key, None)
        return cleaned

    # Provider-private reasoning blocks are valid only on their producing family.
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
        cleaned = scrub_native_custody(messages)
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
        """Strip replayed reasoning once for a non-overflow OpenRouter 400."""
        if _is_structured_context_overflow_exception(exc):
            return None
        if not target.get("supports_openrouter_extensions"):
            return None
        if not self._is_http_status(exc, 400):
            return None
        return self._reroute_same_model_kwargs(target, kwargs)

    @staticmethod
    def _rotate_openrouter_session_affinity(payload: Dict[str, Any]) -> None:
        """A deliberate endpoint reroute must not reuse its sticky session key."""
        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body.get("session_id"):
            return
        previous = str(extra_body["session_id"])
        digest = hashlib.sha256(
            f"{previous}\0reroute\0{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:32]
        extra_body["session_id"] = f"ouroboros-session-{digest}"

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
        model_id = str(kwargs.get("model") or "").strip().lstrip("~")
        preserve_reasoning = (
            allow_portable_reasoning
            and _reasoning_signature_portable_across_or_providers(model_id)
            # OpenAI encrypted-reasoning items are NOT reliably portable across
            # OpenRouter sibling upstreams in the field (2026-07, gpt-5.6-sol on
            # 3x OpenAI + 2x Azure endpoints: "The encrypted content for item
            # rs_... could not be ..." 400s after 429-reroutes killed whole
            # benchmark runs; the 2026-06 replay probe did not cover this mix).
            # openai/* therefore strips on reroute as it did before v6.49.0;
            # preserve stays for Anthropic/Gemini whose signatures verified
            # portable. The proactive continuity pin at dispatch (other callers
            # of the predicate) is intentionally unchanged.
            and not model_id.startswith("openai/")
        )
        if preserve_reasoning:
            retry_kwargs = copy.deepcopy(kwargs)
            self._rotate_openrouter_session_affinity(retry_kwargs)
            return retry_kwargs
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
        self._rotate_openrouter_session_affinity(retry_kwargs)
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
        switched = str(from_model or "").strip() != str(to_model or "").strip()
        has_native_custody = any(
            any(custody_private_key(key) for key in message)
            for message in messages if isinstance(message, dict)
        )
        prepared = scrub_native_custody(messages) if switched and has_native_custody else messages
        if cls._model_family(from_model) == cls._model_family(to_model):
            return prepared
        return cls._strip_openrouter_roundtrip_metadata(prepared)

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
        if not err or _is_structured_context_overflow_body(err):
            return None
        if not self._is_transient_body_error(err):
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
            if self._has_replayed_reasoning_metadata(reroute.get("messages") or [])
            else "dropped",
        )
        return reroute

    def _strip_kwargs_for_encrypted_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Strip replayed encrypted reasoning for a non-overflow body 400."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        body_err = self._provider_body_error(resp_dict)
        if not isinstance(body_err, dict) or _is_structured_context_overflow_body(body_err):
            return None
        try:
            code = int(body_err.get("code") or 0)
        except (TypeError, ValueError):
            code = 0
        if code != 400:
            return None
        if "encrypted content" not in str(body_err.get("message") or "").lower():
            return None
        stripped = self._reroute_same_model_kwargs(target, kwargs)
        if stripped is not None:
            log.warning(
                "OpenRouter strip-and-retry after encrypted-reasoning body error (code=400)"
            )
        return stripped

    def _param_retry_kwargs_for_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        usage_model: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply exception-path parameter recovery to a non-overflow body 400."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        body_err = self._provider_body_error(resp_dict)
        if not isinstance(body_err, dict) or _is_structured_context_overflow_body(body_err):
            return None
        try:
            code = int(body_err.get("code") or 0)
        except (TypeError, ValueError):
            code = 0
        if code != 400:
            return None
        message = str(body_err.get("message") or "")
        if not message:
            return None
        return self._retry_without_optional_sampling(kwargs, usage_model, RuntimeError(message))

    # Anthropic accepts at most four declared cache breakpoints per request.
    _MAX_CACHE_BREAKPOINTS = 4

    @staticmethod
    def _payload_cache_breakpoints(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Blocks carrying a ``cache_control`` marker, in the real wire prefix order
        ``tools -> system -> messages`` — NOT the order arguments happen to arrive in.

        Descends one level INTO a block's own ``content`` list: a direct-Anthropic
        ``tool_result`` block nests its blocks (``_anthropic_messages`` builds it from a
        ``role="tool"`` message whose content is a list), so the sealed transcript anchor
        (``loop.seal_task_transcript``) sits at ``messages[i].content[j].content[k]``.
        Missing it undercounts the cap and leaves that anchor out of TTL ordering exactly
        on the lane whose provider enforces both. ``tool_result`` is the only nested-content
        shape, and the descent is route-independent because no other payload nests."""
        holders: List[Dict[str, Any]] = []
        for key in ("tools", "system", "messages"):
            part = payload.get(key)
            for item in (part if isinstance(part, list) else [part]):
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("cache_control"), dict):
                    holders.append(item)
                content = item.get("content")
                if isinstance(content, list):
                    if is_replayed_native_content(content):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if isinstance(block.get("cache_control"), dict):
                            holders.append(block)
                        nested = block.get("content")
                        if isinstance(nested, list):
                            holders.extend(
                                inner for inner in nested
                                if isinstance(inner, dict)
                                and isinstance(inner.get("cache_control"), dict)
                            )
        return holders

    def _normalize_payload_cache_ttl(
        self,
        target: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Finalize cache policy on the FULLY ASSEMBLED payload; report its strongest TTL.

        The one point where tools, system and messages coexist, hence the single home for
        send-time cache policy (v6.77.0 — replaces two per-builder "mark the last tool"
        copies and restores the TTL ordering guard lost in 176567b BY CONSTRUCTION): a
        ``1h`` breakpoint promotes the earlier EXISTING breakpoints to ``1h`` (a longer TTL
        must precede a shorter one — 5m tools before 1h system is a hard 400) and never
        creates a marker on an earlier segment; a bare marker is the provider default and
        ranks as 5m; the ONLY marker it ever adds is on the last tool schema, and only when
        the tools segment carries none (unconditional on this family in both deleted sites —
        a tool-free payload therefore stays uncached HERE, and system/messages never gain a
        marker they did not declare; a tool-free lane is cached only by DECLARING its stable
        prefix at the caller, as the review surfaces and the safety supervisor do via
        ``review_helpers.cached_prompt_blocks``); above the four-breakpoint cap the four EARLIEST
        (governance-prefix) markers are kept, the tail MARKERS — never content — are dropped
        and the reduction is disclosed in usage (rationale and the builder-side loud layer:
        ``docs/ARCHITECTURE.md``). Only this freshly assembled payload is normalized —
        never caller-owned messages/tools, the canonical transcript, or a route that cannot
        carry these markers (``_route_normalizes_cache_breakpoints``). "1h" wins over
        "default" so pricing bills the extended-tier write multiplier.

        The owner's global TTL (``config.resolve_prompt_cache_ttl``, owner decision
        2026-08-08 Q2=A) has its single WIRE authority here — the one place that decides
        what every marker on this family actually ships as. It is not the only READER:
        ``review_helpers.cached_prompt_blocks(ttl=None)`` projects the same setting into
        the block it owns so a non-normalizing route still carries the owner's tier; on
        this family the finalizer would stamp that block to the same value anyway, so the
        two readers cannot diverge on the wire (``config.resolve_prompt_cache_ttl`` names
        both). When the setting names an explicit tier
        ('5m'/'1h') it is stamped onto EVERY existing breakpoint of this family —
        including caller-declared review/safety prefixes, which is what makes it an
        HONEST override rather than a floor — before the promotion rule runs, so
        ordering stays legal by construction (the 176567b every-call-400 class).
        'default' keeps the pre-setting behavior byte-for-byte: bare markers stay bare
        and a caller-declared ttl stands. It never CREATES a marker (the d32f703d
        empty-block 400 class), and non-Anthropic wire formats are untouched (the
        v5.30.0 Gemini ttl-field class).
        """
        breakpoints = self._payload_cache_breakpoints(payload)
        note: Optional[Dict[str, Any]] = None
        if _route_normalizes_cache_breakpoints(target):
            tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
            if not any(isinstance(t, dict) and isinstance(t.get("cache_control"), dict) for t in tools):
                for tool in reversed(tools):
                    # Schema entries only — skips an appended openrouter:web_search tool.
                    if isinstance(tool, dict) and (
                        isinstance(tool.get("function"), dict)
                        or tool.get("input_schema") is not None
                    ):
                        tool["cache_control"] = {"type": "ephemeral"}
                        breakpoints = self._payload_cache_breakpoints(payload)
                        break
            declared = len(breakpoints)
            if declared > self._MAX_CACHE_BREAKPOINTS:
                for holder in breakpoints[self._MAX_CACHE_BREAKPOINTS:]:
                    holder.pop("cache_control", None)
                breakpoints = breakpoints[:self._MAX_CACHE_BREAKPOINTS]
                note = {"declared": declared, "kept": len(breakpoints),
                        "dropped": declared - len(breakpoints)}
            from ouroboros.config import resolve_prompt_cache_ttl

            global_ttl = resolve_prompt_cache_ttl()
            if global_ttl in _VALID_CACHE_TTLS:
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = global_ttl
            if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = "1h"
        if not hasattr(self, "_cache_breakpoint_tls"):
            self._cache_breakpoint_tls = threading.local()
        self._cache_breakpoint_tls.pending = note
        # Report the strongest APPLIED TTL — the value that flows into usage metadata
        # (llm_usage/llm_round events) and prices the write tier. Readers consume this
        # recorded fact; nothing re-derives an "effective TTL" from the route.
        if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
            return "1h"
        if any(str(b["cache_control"].get("ttl") or "") == "5m" for b in breakpoints):
            return "5m"
        return "default" if breakpoints else None

    def _pop_cache_breakpoint_disclosure(self) -> Optional[Dict[str, Any]]:
        """The pending ≤4-cap reduction record for THIS thread's in-flight call (the
        finalizer writes the slot before every send, so it never mis-attributes)."""
        tls = getattr(self, "_cache_breakpoint_tls", None)
        pending = getattr(tls, "pending", None) if tls is not None else None
        if tls is not None:
            tls.pending = None
        return pending if isinstance(pending, dict) else None

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
        cache_affinity: str = "",
        bypass_response_cache: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call returning (message, usage); no_proxy avoids macOS fork proxy crashes.

        ``response_format`` (e.g. ``{"type": "json_object"}``) is optional request
        intent on the OpenAI-compatible/OpenRouter lanes: local, Anthropic-native,
        and GigaChat routes ignore it, and a provider rejection strips it via the
        optional-parameter retry — callers must keep a text-parse fallback."""
        messages = self._normalize_system_message_placement(messages)
        with capture_attempt_ids() as attempt_ids:
            if use_local:
                message, usage = self._chat_local(
                    messages, tools, max_tokens, tool_choice, timeout=timeout,
                )
            else:
                # Central worker policy: remote calls from worker processes avoid
                # system proxy lookup without every caller remembering a flag.
                no_proxy = no_proxy or in_worker_process()
                target = self._resolve_remote_target(model)
                message, usage = self._chat_remote(
                    target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                    no_proxy=no_proxy,
                    timeout=timeout,
                    allow_server_web_search=allow_server_web_search,
                    response_format=response_format,
                    cache_affinity=cache_affinity,
                    bypass_response_cache=bypass_response_cache,
                )
            usage["ledger_attempt_ids"] = list(attempt_ids)
            return message, usage

    @request_wire_scoped
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
        cache_affinity: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async remote chat; no_proxy keeps forked macOS workers off OS proxy APIs."""
        messages = self._normalize_system_message_placement(messages)
        no_proxy = no_proxy or in_worker_process()
        if tools:
            raise ValueError("chat_async does not support tool calls")
        target = self._resolve_remote_target(model)
        if target.get("provider") == "anthropic":
            with capture_attempt_ids() as attempt_ids:
                result = await asyncio.to_thread(
                    self._chat_anthropic, target, messages, tools, reasoning_effort,
                    max_tokens, tool_choice, temperature, no_proxy, timeout,
                )
            result[1]["ledger_attempt_ids"] = list(attempt_ids)
            return result
        if target.get("provider") == "gigachat":
            # The gigachat library client is synchronous; offload to a thread
            # like the Anthropic path so the event loop is never blocked.
            with capture_attempt_ids() as attempt_ids:
                result = await asyncio.to_thread(
                    self._chat_gigachat, target, messages, tools, reasoning_effort,
                    max_tokens, tool_choice, temperature, no_proxy,
                )
            result[1]["ledger_attempt_ids"] = list(attempt_ids)
            return result
        if no_proxy:
            _oa_client, _http_client = self._make_no_proxy_async_client(target, timeout=timeout)
            try:
                kwargs = self._build_remote_kwargs(
                    target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
                    skip_capability_fetch=True,
                    allow_server_web_search=allow_server_web_search,
                    cache_affinity=cache_affinity,
                )
                prompt_cache_ttl = self._normalize_payload_cache_ttl(target, kwargs)
                with capture_attempt_ids() as attempt_ids:
                    resp = await self._create_chat_completion_with_retries_async(
                        _oa_client.chat.completions.create, kwargs, target,
                    )
                result = self._normalize_remote_response(
                    resp.model_dump(),
                    target,
                    skip_cost_fetch=True,
                    prompt_cache_ttl=prompt_cache_ttl,
                )
                result[1]["ledger_attempt_ids"] = list(attempt_ids)
                return result
            finally:
                try:
                    await _http_client.aclose()
                except Exception:
                    pass
        client = self._get_async_remote_client(target)
        kwargs = self._build_remote_kwargs(
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
            allow_server_web_search=allow_server_web_search,
            cache_affinity=cache_affinity,
        )
        if timeout and timeout > 0:
            # Cached clients are built without a timeout; honor the caller's
            # per-request timeout instead of silently using the SDK default.
            kwargs["timeout"] = float(timeout)
        prompt_cache_ttl = self._normalize_payload_cache_ttl(target, kwargs)
        with capture_attempt_ids() as attempt_ids:
            resp = await self._create_chat_completion_with_retries_async(
                client.chat.completions.create, kwargs, target,
            )
        result = self._normalize_remote_response(
            resp.model_dump(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )
        result[1]["ledger_attempt_ids"] = list(attempt_ids)
        return result

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
        # Local llama.cpp has no vision; avoid flattening base64 into the prompt.
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
            kwargs["timeout"] = float(timeout)

        candidate = _physical_candidate(kwargs)
        local_target = {"provider": "local", "usage_model": "local-model"}
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                request = _attempt_request(local_target, candidate, source="llm.local")
                resp = _execute_candidate(
                    request,
                    lambda: client.chat.completions.create(**candidate),
                    _candidate_before_dispatch(candidate, request),
                )
                last_exc = None
                break
            except UsageAccountingError:
                raise
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                if (_is_structured_context_overflow_exception(exc)
                        or context_overflow_message(err)):
                    raise LocalContextTooLargeError(err) from exc
                # Exception-owned capture proves this attempt; prior ContextVar may be unrelated.
                capture = getattr(exc, "physical_attempt_capture", None)
                if isinstance(capture, PhysicalAttemptCapture) and capture.state in {"dispatched", "unresolved"}:
                    raise  # Outer custody owns an unknown physical outcome.
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
        usage["cost_final"] = True
        return msg, usage

    @staticmethod
    def _strip_reasoning_wrappers(text: str):
        """Strip leading think/reasoning wrappers before the first tool markup."""
        return strip_reasoning_wrappers(text)

    @staticmethod
    def _parse_tool_calls_from_content(
        msg: Dict[str, Any],
        allowed_tool_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Parse local <tool_call> XML or well-formed DeepSeek DSML content."""
        return parse_tool_calls_from_content(msg, allowed_tool_names)

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
                        _ttl = str(block["cache_control"].get("ttl") or "")
                        normalized["cache_control"] = (
                            {"type": "ephemeral", "ttl": _ttl}
                            if _ttl in _VALID_CACHE_TTLS
                            else {"type": "ephemeral"}
                        )
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
                    _ttl = str(block["cache_control"].get("ttl") or "")
                    normalized["cache_control"] = (
                        {"type": "ephemeral", "ttl": _ttl}
                        if _ttl in _VALID_CACHE_TTLS
                        else {"type": "ephemeral"}
                    )
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
        target: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        messages = self._normalize_system_message_placement(messages)
        system_blocks: List[Dict[str, Any]] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for message_index, msg in enumerate(messages):
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
                matching_results = []
                cursor = message_index + 1
                while cursor < len(messages) and str(messages[cursor].get("role") or "") == "tool":
                    matching_results.append(str(messages[cursor].get("tool_call_id") or ""))
                    cursor += 1
                assistant_blocks = (
                    native_content_for_replay(msg, target, matching_results)
                    if target is not None else None
                )
                if assistant_blocks is None:
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
        return anthropic_tools

    @staticmethod
    def _sanitize_chat_completion_tools(
        tools: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        from ouroboros.openai_chat_dispatch import sanitize_function_tools

        def _warn(reason: str, name: str) -> None:
            log.warning("Dropping %s tool schema name: %s", reason, name)

        return sanitize_function_tools(
            tools,
            description_normalizer=LLMClient._stringify_tool_description,
            on_drop=_warn,
        )

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
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
            return None
        if not isinstance(tool_choice, str):
            return None
        if tool_choice in {"required", "any"}:
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _cache_write_split(raw_usage: Dict[str, Any]) -> Dict[str, int]:
        """Anthropic's per-tier cache-write counters, when the provider reports them.

        With the extended (1h) tier live, ``usage.cache_creation`` splits
        ``cache_creation_input_tokens`` into ``ephemeral_5m_input_tokens`` /
        ``ephemeral_1h_input_tokens`` — a 1h request can legitimately produce BOTH
        (e.g. a server-tool block cached at the default tier beside the 1h prefix),
        and pricing must bill only the genuine 1h share at the extended ratio.
        Empty dict when the provider reported no split (older shapes) — the caller
        then bills every write at the reported tier, never a loosened ratio.
        """
        split = raw_usage.get("cache_creation") if isinstance(raw_usage, dict) else None
        if not isinstance(split, dict):
            return {}
        out: Dict[str, int] = {}
        for tier, key in (("5m", "ephemeral_5m_input_tokens"), ("1h", "ephemeral_1h_input_tokens")):
            try:
                value = int(split.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                out[tier] = value
        return out

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
            # v6.77.0: Anthropic EXCLUDES cache reads/writes from `input_tokens`, while
            # `prompt_tokens` is the OpenAI-semantics TOTAL input every consumer assumes —
            # `pricing.regular_input = prompt_tokens - cached - cache_write` clamped fresh
            # input to 0 on a cache-heavy call (and cache_hit_rate could exceed 1.0).
            "prompt_tokens": (
                int(raw_usage.get("input_tokens") or 0)
                + int(raw_usage.get("cache_read_input_tokens") or 0)
                + int(raw_usage.get("cache_creation_input_tokens") or 0)
            ),
            "completion_tokens": int(raw_usage.get("output_tokens") or 0),
            "cached_tokens": int(raw_usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_creation_input_tokens") or 0),
            "provider": "anthropic",
            "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or ""),
        }
        if prompt_cache_ttl:
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        write_split = self._cache_write_split(raw_usage)
        if write_split:
            usage["cache_write_tokens_by_ttl"] = write_split
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            from ouroboros.pricing import estimate_cost_optional

            estimated_cost = estimate_cost_optional(
                usage["resolved_model"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cache_usage={
                    "cached_tokens": usage["cached_tokens"],
                    "cache_write_tokens": usage["cache_write_tokens"],
                    "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
                    "cache_write_tokens_by_ttl": write_split or None,
                },
                provider="anthropic",
            )
            if estimated_cost is not None:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True
        if usage.get("cost") is None:
            usage["cost"] = None
        usage["cost_final"] = bool(
            usage.get("cost") is not None and not usage.get("cost_estimated")
        )
        # Preserve any legacy diagnostic disclosure already staged by a compatibility
        # caller; normal dispatch adaptation is disclosed through usage.request_wire.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note
        _cache_note = self._pop_cache_breakpoint_disclosure()
        if _cache_note:
            usage["prompt_cache_breakpoints_reduced"] = _cache_note

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
        message = retain_native_assistant_content(message, content_blocks, target)
        if tool_calls or str(message.get("content") or "").strip():
            message = mark_replayed_receipts_consumed(message)
        finalize_wire_response(message, usage)
        return message, usage

    @request_wire_scoped
    @anthropic_replay_scoped
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

        system, anthropic_messages = self._build_anthropic_messages(messages, target)
        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        # Modern Anthropic uses adaptive thinking plus output_config.effort.
        _eff = normalize_reasoning_effort(reasoning_effort)
        if _eff == "none":
            payload["thinking"] = {"type": "disabled"}
        elif _eff:
            payload["thinking"] = {"type": "adaptive"}
            # Anthropic has no "minimal" effort; map it to the provider floor.
            payload["output_config"] = {"effort": "low" if _eff == "minimal" else _eff}
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        anthropic_tools = self._build_anthropic_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            anthropic_tool_choice = self._build_anthropic_tool_choice(tool_choice)
            if anthropic_tool_choice:
                payload["tool_choice"] = anthropic_tool_choice
        prompt_cache_ttl = self._normalize_payload_cache_ttl(target, payload)

        url = f"{str(target.get('base_url') or '').rstrip('/')}/messages"
        headers = {
            "x-api-key": str(target.get("api_key") or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        request_timeout = float(timeout) if timeout and timeout > 0 else 120

        def _send(candidate: Dict[str, Any]):
            candidate = _physical_candidate(candidate)
            candidate = prepare_wire_payload_for_send(
                target, candidate, api_surface="messages",
            )
            request = _attempt_request(target, candidate, source="llm.anthropic")

            def _post():
                if no_proxy:
                    # Build a session with proxy detection disabled for macOS fork-safety.
                    with requests.Session() as session:
                        session.trust_env = False
                        sent = session.post(url, headers=headers, json=candidate, timeout=request_timeout)
                else:
                    sent = requests.post(url, headers=headers, json=candidate, timeout=request_timeout)
                if sent.status_code >= 400:
                    body_preview = (sent.text or "")[:2000]
                    raise requests.HTTPError(
                        f"{sent.status_code} {sent.reason} for url {sent.url}: {body_preview}",
                        response=sent,
                    )
                return sent

            try:
                result = _execute_candidate(
                    request,
                    _post,
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                return result
            except UsageAccountingError:
                # Central UAE discard, driver parity (triad r4).
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        try:
            response = _send(payload)
        except UsageAccountingError:
            raise  # _send already discarded any pending clamp note (triad r4)
        except Exception as exc:
            retry_payload = plan_next_wire_retry(payload, error=exc)
            if retry_payload is None:
                self._pop_effort_clamp_disclosure()
                raise
            for _ in range(8):
                try:
                    response = _send(retry_payload)
                    break
                except UsageAccountingError:
                    raise
                except Exception as retry_exc:
                    retry_payload = plan_next_wire_retry(
                        retry_payload, error=retry_exc,
                    )
                    if retry_payload is None:
                        self._pop_effort_clamp_disclosure()
                        raise
            else:
                self._pop_effort_clamp_disclosure()
                raise RuntimeError("request-wire recovery action bound exhausted") from exc
        return self._normalize_anthropic_response(
            response.json(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )

    # ------------------------------------------------------------------
    # GigaChat (native `gigachat` library — NOT OpenAI-compatible)
    # ------------------------------------------------------------------
    @staticmethod
    def _new_gigachat_client(
        target: Dict[str, Any],
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """Build a GigaChat library client for the given target."""
        try:
            from gigachat import GigaChat
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "The 'gigachat' package is required to use gigachat:: models. "
                "Install it with: pip install gigachat"
            ) from exc
        kwargs: Dict[str, Any] = {
            "scope": str(target.get("scope") or "GIGACHAT_API_PERS"),
            "verify_ssl_certs": bool(target.get("verify_ssl_certs", True)),
        }
        for source, destination in (
            ("api_key", "credentials"), ("user", "user"), ("password", "password"),
            ("base_url", "base_url"),
        ):
            value = str(target.get(source) or "")
            # Provider Test carries an explicit access-token field to suppress
            # inherited auth.  Its empty credential is equally authoritative:
            # omitting it would let the library reload GIGACHAT_CREDENTIALS.
            if value or (source == "api_key" and "access_token" in target):
                kwargs[destination] = value
        if "access_token" in target:
            kwargs["access_token"] = str(target.get("access_token") or "")
        if timeout and timeout > 0:
            kwargs["timeout"] = float(timeout)
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return GigaChat(**kwargs)

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

        if cache_key not in self._gigachat_clients:
            self._gigachat_clients[cache_key] = self._new_gigachat_client(target, timeout=timeout)
        return self._gigachat_clients[cache_key]

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

        candidate = _physical_candidate(payload)
        request = _attempt_request(target, candidate, source="llm.gigachat")
        completion = _execute_candidate(
            request,
            lambda: client.chat(candidate),
            _candidate_before_dispatch(candidate, request),
        )
        return self._normalize_gigachat_response(completion, target)

    def _normalize_gigachat_response(
        self,
        completion: Any,
        target: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a GigaChat ``ChatCompletion`` into (message, usage) dicts.

        A GigaChat ``function_call`` becomes a single OpenAI-style ``tool_calls``
        entry (arguments re-encoded as a JSON string). GigaChat exposes no
        automatic cost source, so the normalized usage reports ``cost=None``.
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
            "cost": None,
            "cost_final": False,
        }

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
        cache_affinity: str = "",
        bypass_response_cache: bool = False,
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
        # Official direct OpenAI Chat uses the current completion-token carrier:
        # provider-wide; model names are not capability authority across routes.
        direct_openai = provider == "openai"
        token_limit_key = "max_completion_tokens" if direct_openai else "max_tokens"
        if not target.get("supports_openrouter_extensions"):
            prepared_tools = [
                {k: v for k, v in tool.items() if k != "cache_control"}
                for tool in self._sanitize_chat_completion_tools(tools)
            ]
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
            if provider == "openai":
                cache_identity = self._prompt_cache_identity(
                    str(target.get("usage_model") or resolved_model),
                    clean_messages,
                )
                if cache_identity:
                    # OpenAI's named affinity key keeps requests sharing the
                    # stable governance prefix on the same cache bucket.
                    kwargs["prompt_cache_key"] = cache_identity
            requested_effort = normalize_reasoning_effort(reasoning_effort)
            if direct_openai:
                # Direct-OpenAI route honors the configured OUROBOROS_EFFORT_*
                # lanes instead of silently dropping them (OpenRouter parity).
                # Exact-route request-wire evidence, not legacy model-global
                # rows, owns any provider-required adaptation after this build.
                kwargs["reasoning_effort"] = requested_effort
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format:
                kwargs["response_format"] = dict(response_format)
            if prepared_tools:
                kwargs["tools"] = prepared_tools
                kwargs["tool_choice"] = tool_choice
            if bypass_response_cache and provider == "openai-compatible":
                # Must ride in extra_body: the OpenAI SDK rejects unknown top-level
                # kwargs with TypeError, so a raw `cache=` argument never reaches
                # the wire.
                _eb = kwargs.setdefault("extra_body", {})
                if isinstance(_eb, dict):
                    _eb["cache"] = {"no-cache": True}
            return kwargs

        effort = normalize_reasoning_effort(reasoning_effort)
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
        cache_identity = self._explicit_cache_affinity_identity(
            str(target.get("usage_model") or resolved_model),
            cache_affinity,
        ) or self._openrouter_session_identity(
            str(target.get("usage_model") or resolved_model),
            messages,
        )
        if cache_identity:
            # The OpenAI SDK forwards extra_body members as top-level
            # OpenRouter request fields; session_id provides sticky routing.
            extra_body["session_id"] = cache_identity

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
                allow_cache_ttl=cache_model.startswith("anthropic/"),
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
            # Tool cache markers are placed once, at the send-time payload finalizer
            # (`_normalize_payload_cache_ttl`) — it is the only point that sees tools,
            # system and messages together and can order their TTLs.
            kwargs["tools"] = prepared_tools
            kwargs["tool_choice"] = tool_choice

        # With require_parameters, unsupported params cause OpenRouter 404s.
        # Unknown capabilities mean no stripping.
        if skip_capability_fetch:
            # "Skip" means skip the NETWORK fetch (no_proxy fork-safety), not
            # ignore an already-warm capability cache: a worker forked after the
            # one-shot /models fetch still proactively strips unsupported params
            # instead of paying a reactive 404 + retry on every reviewer call.
            supported = (
                self._SUPPORTED_PARAMS_CACHE.get(resolved_model)
                if self._SUPPORTED_PARAMS_FETCHED
                else None
            )
        else:
            supported = self._get_supported_parameters(resolved_model)
        if supported is not None:
            unsupported = [
                optional_param for optional_param in _OPTIONAL_DROPPABLE_PARAMS
                if optional_param not in supported and optional_param in kwargs
            ]
            note_provider_metadata_drop_fields(unsupported)
        return kwargs

    def _normalize_remote_response(
        self,
        resp_dict: Dict[str, Any],
        target: Dict[str, Any],
        skip_cost_fetch: bool = False,
        prompt_cache_ttl: Optional[str] = None,
        wire_completion: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize an OpenAI-compatible response; skip_cost_fetch keeps no_proxy pure."""
        usage = resp_dict.get("usage") or {}
        if isinstance(usage, dict):
            # These keys are host-owned projections of designated outer fields;
            # provider usage extensions must not spoof their provenance.
            usage.pop("response_finish_reason", None)
            usage.pop("response_provider", None)
        # An HTTP-200 that carried a provider body-error (OpenRouter passes
        # 429/5xx through the body) reaches here only when a same-model reroute
        # was unavailable or also errored. Surface it as a typed marker so the
        # caller classifies it as a real rate_limit/provider_transient instead of
        # a blank finish_reason=null "incomplete response".
        _body_err = self._provider_body_error(resp_dict)
        if _body_err:
            usage["provider_error"] = {
                "code": _body_err.get("code"),
                "type": _body_err.get("type"),
                "message": str(_body_err.get("message") or "")[:300],
                "kind": "rate_limit" if self._is_transient_body_error(_body_err) and str(_body_err.get("code")) == "429"
                else ("provider_transient" if self._is_transient_body_error(_body_err) else "provider_error"),
            }
        choices = resp_dict.get("choices") or [{}]
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        msg = dict(first_choice.get("message") or {})
        # ``finish_reason`` belongs to the response choice, not the assistant
        # message. Preserve it as an observational usage fact so diagnostics can
        # distinguish a provider-selected stop/length from an absent marker. Do
        # not put it into canonical history: strict providers reject response-only
        # fields when that history is replayed.
        if "finish_reason" in first_choice:
            usage["response_finish_reason"] = _bounded_response_metadata_label(
                first_choice.get("finish_reason"),
            )
        response_provider = _bounded_response_metadata_label(resp_dict.get("provider"))
        if response_provider is not None:
            # This optional upstream serving label never replaces the accounting
            # provider assigned below.
            usage["response_provider"] = response_provider
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
            if usage.get("cost") is None:
                gen_id = resp_dict.get("id") or ""
                if gen_id:
                    cost = self._fetch_generation_cost(gen_id, target)
                    if cost is not None:
                        usage["cost"] = cost

        usage["provider"] = str(target.get("provider") or "openrouter")
        usage["resolved_model"] = str(target.get("usage_model") or target.get("resolved_model") or "")
        if prompt_cache_ttl and not usage.get("prompt_cache_ttl"):
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        # Anthropic's per-tier write split, when the route passed it through.
        _write_split = self._cache_write_split(usage)
        if _write_split and not usage.get("cache_write_tokens_by_ttl"):
            usage["cache_write_tokens_by_ttl"] = _write_split
        if usage.get("cost") is None and (usage.get("prompt_tokens") or usage.get("completion_tokens")):
            from ouroboros.pricing import estimate_cost_optional

            estimated_cost = estimate_cost_optional(
                usage["resolved_model"],
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                cache_usage={
                    "cached_tokens": int(usage.get("cached_tokens") or 0),
                    "cache_write_tokens": int(usage.get("cache_write_tokens") or 0),
                    "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
                    "cache_write_tokens_by_ttl": (
                        usage.get("cache_write_tokens_by_ttl")
                        if isinstance(usage.get("cache_write_tokens_by_ttl"), dict)
                        else None
                    ),
                },
                allow_live_fetch=not skip_cost_fetch,
                provider=usage["provider"],
            )
            if estimated_cost is not None:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True
        if usage.get("cost") is None:
            usage["cost"] = None
        usage["cost_final"] = bool(
            usage.get("cost") is not None and not usage.get("cost_estimated")
        )
        # Preserve any legacy diagnostic disclosure already staged by a compatibility
        # caller; normal dispatch adaptation is disclosed through usage.request_wire.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note
        # Same disclosure norm for a ≤4-cap cache-marker reduction (v6.77.0): never silent.
        _cache_note = self._pop_cache_breakpoint_disclosure()
        if _cache_note:
            usage["prompt_cache_breakpoints_reduced"] = _cache_note
        from ouroboros.openai_chat_dispatch import normalize_direct_openai_completion

        msg, usage = normalize_direct_openai_completion(msg, usage, wire_completion)
        _custom_receipts = usage.get("_request_wire_custom_receipts", ())
        finalize_wire_response(msg, usage, custom_receipts=_custom_receipts)
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

    @request_wire_scoped
    def _create_chat_completion_with_retries(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        def _send(candidate: Dict[str, Any]) -> Any:
            candidate = _physical_candidate(candidate)
            candidate = prepare_wire_payload_for_send(
                target, candidate, api_surface="chat.completions",
            )
            request = _attempt_request(target, candidate)
            try:
                result = _execute_candidate(
                    request,
                    lambda: create_fn(**candidate),
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                return result
            except UsageAccountingError:
                # Admission failure cannot leave its disclosure for a later call.
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        def _body_error(response: Any) -> Optional[Dict[str, Any]]:
            try:
                return self._provider_body_error(response.model_dump())
            except Exception:
                return None

        def _recover_existing(
            candidate: Dict[str, Any],
            *,
            failure: Optional[Exception] = None,
            response: Any = None,
        ) -> Any:
            """One bounded exception/body state machine, then signature recovery."""
            try:
                current_candidate = candidate
                current_failure = failure
                current_response = response
                signature_used = False
                for _ in range(8):
                    if current_failure is None:
                        body = _body_error(current_response)
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=body, body_error=True,
                        )
                        if retry_kwargs is None:
                            return current_response
                    else:
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=current_failure,
                        )
                        if retry_kwargs is None and not signature_used:
                            retry_kwargs = self._openrouter_signature_retry_kwargs(
                                target, current_candidate, current_failure,
                            )
                            signature_used = retry_kwargs is not None
                        if retry_kwargs is None:
                            raise current_failure
                    current_candidate = retry_kwargs
                    try:
                        current_response = _send(retry_kwargs)
                        current_failure = None
                    except UsageAccountingError:
                        raise
                    except Exception as retry_exc:
                        current_failure = retry_exc
                        current_response = None
                if current_failure is not None:
                    raise current_failure
                return current_response
            except Exception:
                # The recovery ladder died terminally: discard any pending
                # effort-clamp note (e.g. the floored learning retry's
                # learned_floor disclosure) so it cannot misattach to a later,
                # unrelated response on this thread (plan-review r3; lanes that
                # never call _clamp_effort_for_model at build time would not
                # reset it).
                self._pop_effort_clamp_disclosure()
                raise

        try:
            resp = _send(kwargs)
        except UsageAccountingError:
            raise  # _send already discarded any pending clamp note (triad r4)
        except Exception as exc:
            cache_retry_kwargs = self._retry_without_prompt_cache_parameter(kwargs, target, exc)
            if cache_retry_kwargs is not None:
                try:
                    resp = _send(cache_retry_kwargs)
                    kwargs = cache_retry_kwargs
                except UsageAccountingError:
                    raise
                except Exception as cache_retry_exc:
                    return _recover_existing(
                        cache_retry_kwargs, failure=cache_retry_exc,
                    )
            else:
                return _recover_existing(kwargs, failure=exc)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                resp = _send(reroute_kwargs)
            except UsageAccountingError:
                raise
            except Exception:
                return resp
            kwargs = reroute_kwargs
        # An encrypted-reasoning 400 delivered in the body (directly, or on the
        # response of the reroute above) gets the same one-shot strip-and-retry
        # as the exception path — never a permanent task-killing bad_request.
        strip_kwargs = self._strip_kwargs_for_encrypted_body_error(resp, kwargs, target)
        if strip_kwargs is not None:
            try:
                resp = _send(strip_kwargs)
                kwargs = strip_kwargs
            except UsageAccountingError:
                raise
            except Exception:
                return resp
        return _recover_existing(kwargs, response=resp)

    @request_wire_scoped
    async def _create_chat_completion_with_retries_async(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        async def _send(candidate: Dict[str, Any]) -> Any:
            candidate = _physical_candidate(candidate)
            candidate = prepare_wire_payload_for_send(
                target, candidate, api_surface="chat.completions",
            )
            request = _attempt_request(target, candidate)
            try:
                result = await _execute_candidate_async(
                    request,
                    lambda: create_fn(**candidate),
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                return result
            except UsageAccountingError:
                # Sync-driver parity: central UAE discard (triad r4).
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        def _body_error(response: Any) -> Optional[Dict[str, Any]]:
            try:
                return self._provider_body_error(response.model_dump())
            except Exception:
                return None

        async def _recover_existing(
            candidate: Dict[str, Any],
            *,
            failure: Optional[Exception] = None,
            response: Any = None,
        ) -> Any:
            """Async twin of the bounded exception/body state machine."""
            try:
                current_candidate = candidate
                current_failure = failure
                current_response = response
                signature_used = False
                for _ in range(8):
                    if current_failure is None:
                        body = _body_error(current_response)
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=body, body_error=True,
                        )
                        if retry_kwargs is None:
                            return current_response
                    else:
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=current_failure,
                        )
                        if retry_kwargs is None and not signature_used:
                            retry_kwargs = self._openrouter_signature_retry_kwargs(
                                target, current_candidate, current_failure,
                            )
                            signature_used = retry_kwargs is not None
                        if retry_kwargs is None:
                            raise current_failure
                    current_candidate = retry_kwargs
                    try:
                        current_response = await _send(retry_kwargs)
                        current_failure = None
                    except UsageAccountingError:
                        raise
                    except Exception as retry_exc:
                        current_failure = retry_exc
                        current_response = None
                if current_failure is not None:
                    raise current_failure
                return current_response
            except Exception:
                self._pop_effort_clamp_disclosure()
                raise

        try:
            resp = await _send(kwargs)
        except UsageAccountingError:
            raise  # _send already discarded any pending clamp note (triad r4)
        except Exception as exc:
            cache_retry_kwargs = self._retry_without_prompt_cache_parameter(kwargs, target, exc)
            if cache_retry_kwargs is not None:
                try:
                    resp = await _send(cache_retry_kwargs)
                    kwargs = cache_retry_kwargs
                except UsageAccountingError:
                    raise
                except Exception as cache_retry_exc:
                    return await _recover_existing(
                        cache_retry_kwargs, failure=cache_retry_exc,
                    )
            else:
                return await _recover_existing(kwargs, failure=exc)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                resp = await _send(reroute_kwargs)
            except UsageAccountingError:
                raise
            except Exception:
                return resp
            kwargs = reroute_kwargs
        # An encrypted-reasoning 400 delivered in the body (directly, or on the
        # response of the reroute above) gets the same one-shot strip-and-retry
        # as the exception path — never a permanent task-killing bad_request.
        strip_kwargs = self._strip_kwargs_for_encrypted_body_error(resp, kwargs, target)
        if strip_kwargs is not None:
            try:
                resp = await _send(strip_kwargs)
                kwargs = strip_kwargs
            except UsageAccountingError:
                raise
            except Exception:
                return resp
        return await _recover_existing(kwargs, response=resp)

    @request_wire_scoped
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
        cache_affinity: str = "",
        bypass_response_cache: bool = False,
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
                    cache_affinity=cache_affinity,
                    bypass_response_cache=bypass_response_cache,
                )
                prompt_cache_ttl = self._normalize_payload_cache_ttl(target, kwargs)
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
                    wire_completion=resp,
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
            cache_affinity=cache_affinity,
            bypass_response_cache=bypass_response_cache,
        )
        if timeout and timeout > 0:
            # Cached clients are built without a timeout; honor the caller's
            # per-request timeout instead of silently using the SDK default.
            kwargs["timeout"] = float(timeout)
        prompt_cache_ttl = self._normalize_payload_cache_ttl(target, kwargs)
        resp = self._create_chat_completion_with_retries(
            client.chat.completions.create,
            kwargs,
            target,
        )
        return self._normalize_remote_response(
            resp.model_dump(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
            wire_completion=resp,
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
        return os.environ.get("OUROBOROS_MODEL", OPENROUTER_DEFAULTS["main"])

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = self.default_model()
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if light and light != main:
            models.append(light)
        return models


def openrouter_web_search_server_tool(
    *,
    api_key: str,
    model: str,
    query: str,
    search_context_size: str,
    accounting_scope: Optional[UsageScope] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Run OpenRouter's provider-owned web_search server tool."""

    from ouroboros.net_transport import web_search_openai_client

    client = web_search_openai_client(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout,
        default_headers=dict(OPENROUTER_APP_HEADERS),
    )
    payload = dict(
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
    candidate = _physical_candidate(payload)
    request = _attempt_request(
        {"provider": "openrouter", "usage_model": model, "resolved_model": model},
        candidate,
        source="web_search.openrouter",
    )
    before_dispatch = _candidate_before_dispatch(candidate, request)
    if accounting_scope is None:
        return _execute_candidate(
            request, lambda: client.chat.completions.create(**candidate), before_dispatch,
        )
    with usage_scope(accounting_scope):
        return _execute_candidate(
            request, lambda: client.chat.completions.create(**candidate), before_dispatch,
        )


def anthropic_web_search_server_tool(
    *,
    api_key: str,
    model: str,
    query: str,
    accounting_scope: Optional[UsageScope] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Run Anthropic's provider-owned web_search server tool."""

    import anthropic

    client_kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if timeout is not None:
        client_kwargs["timeout"] = float(timeout)
    client = anthropic.Anthropic(**client_kwargs)
    payload = dict(
        model=model,
        max_tokens=2048,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": query}],
    )
    candidate = _physical_candidate(payload)
    request = _attempt_request(
        {"provider": "anthropic", "usage_model": model, "resolved_model": model},
        candidate,
        source="web_search.anthropic",
    )
    before_dispatch = _candidate_before_dispatch(candidate, request)
    if accounting_scope is None:
        return _execute_candidate(
            request, lambda: client.messages.create(**candidate), before_dispatch,
        )
    with usage_scope(accounting_scope):
        return _execute_candidate(
            request, lambda: client.messages.create(**candidate), before_dispatch,
        )
