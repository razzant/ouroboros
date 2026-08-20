"""The OpenAI-compatible request and response projection.

Every non-native route — OpenRouter, direct OpenAI, cloud.ru, MiniMax, a vLLM
server — speaks the OpenAI chat-completions shape, and the differences between
them are request options: which token-limit key, which reasoning carrier, which
cache affinity, which provider routing block. This module owns building that
payload and reading the response back into the normalized ``(message, usage)``
every caller consumes.
"""


from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.llm_attempt import supports_message_cache_control
from ouroboros.llm_capability_policy import (
    _OPTIONAL_DROPPABLE_PARAMS,
    normalize_reasoning_effort,
)
from ouroboros.llm_messages import _reasoning_signature_portable_across_or_providers
from ouroboros.llm_routing import _resolve_or_provider


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


_FALSE_LIKE_ENV_VALUES = {"", "0", "false", "no", "off"}


class _OpenAICompatibleLaneMixin:
    """OpenAI-compatible payload assembly and response normalisation."""

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
            if provider == "openai":
                cache_identity = self._prompt_cache_identity(
                    str(target.get("usage_model") or resolved_model),
                    clean_messages,
                )
                if cache_identity:
                    # OpenAI's named affinity key keeps requests sharing the
                    # stable governance prefix on the same cache bucket.
                    kwargs["prompt_cache_key"] = cache_identity
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
            if bypass_response_cache and provider == "openai-compatible":
                # Must ride in extra_body: the OpenAI SDK rejects unknown top-level
                # kwargs with TypeError, so a raw `cache=` argument never reaches
                # the wire.
                _eb = kwargs.setdefault("extra_body", {})
                if isinstance(_eb, dict):
                    _eb["cache"] = {"no-cache": True}
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
        self._apply_rejected_param_cache(kwargs, resolved_model)
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
                "type": _body_err.get("type"),
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
        # v6.61.1 (Q7 disclosure): a learned-ceiling clamp recorded at payload build
        # (_build_remote_kwargs → _clamp_effort_for_model) rides THIS call's usage —
        # covers both the OpenRouter and the OpenAI-compatible direct lanes.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note
        # Same disclosure norm for a ≤4-cap cache-marker reduction (v6.77.0): never silent.
        _cache_note = self._pop_cache_breakpoint_disclosure()
        if _cache_note:
            usage["prompt_cache_breakpoints_reduced"] = _cache_note

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
