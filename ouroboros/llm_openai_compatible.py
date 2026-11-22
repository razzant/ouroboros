"""The OpenAI-compatible request and response projection.

Every non-native route — OpenRouter, direct OpenAI, cloud.ru, MiniMax, a vLLM
server — speaks the OpenAI chat-completions shape, and the differences between
them are request options: which token-limit key, which reasoning carrier, which
cache affinity, which provider routing block — and, for OpenAI's public API, which
part of the leading system message its whole-section prompt cache may see
(``_project_openai_family_system``). This module owns building that payload and
reading the response back into the normalized ``(message, usage)`` every caller
consumes.
"""


from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple


from ouroboros.llm_attempt import (
    _applied_payload_cache_ttl, apply_processing_preference, attach_processing_receipt, body_error_facts,
    openai_family_route, supports_message_cache_control,
)
from ouroboros.llm_messages import project_declared_system_prefix
from ouroboros.openai_responses import (
    REPLY_FACT_KEYS, attach_reply_facts, completion_view, responses_payload, responses_route,
)
from ouroboros.usage_accounting import UsageScope, usage_scope
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from ouroboros.opencode_go import go_chat_headers, is_go_chat_route
from ouroboros.llm_capability_policy import (
    _EFFORT_CLAMP_CVAR,
    _OPTIONAL_DROPPABLE_PARAMS,
    normalize_reasoning_effort,
)
from ouroboros.llm_routing import _resolve_or_provider, openrouter_chat_routing
from ouroboros.provider_models import (
    normalize_deepseek_reasoning_effort,
    normalize_minimax_reasoning_effort,
    normalize_zai_reasoning_effort,
)
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    finalize_wire_response,
    note_provider_metadata_drop_fields,
)
from ouroboros.utils import sanitize_tool_result_for_log
from ouroboros.config import runtime_setting
from ouroboros._usage_response import (
    _reported_token_count, cache_pricing_usage, cache_read_count, cache_write_total, malformed_token_counts,
    normalize_go_price, observed_processing_mode, response_reported_price, text_token_counts,
)
from ouroboros.event_usage import REPORTED_TOKENS_KEY, event_token_counts


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


_FALSE_LIKE_ENV_VALUES = {"", "0", "false", "no", "off"}

# Response-only labels are diagnostic facts, not canonical assistant fields. Keep
# provider-supplied values bounded and printable before they enter usage custody.
_RESPONSE_METADATA_LABEL_MAX_CHARS = 160


def _project_openai_family_system(target: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The OpenAI-family send copy of a declared leading system message.

    Without explicit breakpoints OpenAI looks a cache up only at the end of the leading
    system group (``llm_attempt.openai_family_model``): keep only the builder-declared stable
    governance blocks there and carry the mutable context as a host notice before the task,
    so a new conversation is served the prefix its predecessors already cached. OpenRouter
    turns the block markers into explicit breakpoints for this family, so there each marked
    leading block also stays its own system item and is reused independently. Runs before
    the direct/OpenRouter branch split and the marker strip, so direct OpenAI, OpenRouter
    and the prospective wrap-up candidate project from one function; the canonical
    transcript keeps its single system message. The shape that was sent rides
    ``target["wire_layout"]`` (per-call, never a thread-local) into ``usage`` in
    ``_normalize_remote_response``.
    """
    if not openai_family_route(target):
        return messages
    keep_marked = bool(target.get("supports_openrouter_extensions")) and supports_message_cache_control(
        str(target.get("resolved_model") or ""))
    return project_declared_system_prefix(target, messages, keep_marked=keep_marked)


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


def _stage_effort_clamp_disclosure(
    requested: str, applied: str, model: str, *, reason: str = "provider_wire_mapping",
) -> None:
    """Stage this call's requested/applied effort note for ``_pop_effort_clamp_disclosure``.

    A tier the provider dialect changed is disclosed; an unchanged one stages ``None``, so a
    stale note from an earlier call is never inherited.
    """
    _EFFORT_CLAMP_CVAR.set(
        None if applied == requested
        else {"requested": requested, "applied": applied, "reason": reason, "model": model}
    )


class _OpenAICompatibleLaneMixin:
    """OpenAI-compatible payload assembly and response normalisation."""

    @staticmethod
    def _openrouter_main_web_search_tool() -> Optional[Dict[str, Any]]:
        mode = str(runtime_setting("OUROBOROS_MAIN_WEB_SEARCH") or "off").strip().lower()
        if mode not in {"openrouter", "openrouter_server", "server", "on", "true", "1"}:
            return None
        engine = str(runtime_setting("OUROBOROS_MAIN_WEB_SEARCH_ENGINE") or "auto").strip() or "auto"
        parameters: Dict[str, Any] = {}
        if engine != "auto":
            parameters["engine"] = engine
        try:
            max_total = int(runtime_setting("OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS", "") or 0)
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
        stream: bool = False,
    ) -> Dict[str, Any]:
        target["requested_reasoning_effort"] = normalize_reasoning_effort(reasoning_effort)
        messages = self._normalize_system_message_placement(messages)
        resolved_model = str(target.get("resolved_model") or "")
        provider = str(target.get("provider") or "")
        # Image blocks are encoded as received: whether this route gets pixels,
        # a caption or a marker was decided by the send policy (vision_routing),
        # never by this builder from a model name.
        messages = _project_openai_family_system(target, messages)
        if responses_route(target):  # the connection's own API: the same options in a Responses body
            return self._responses_kwargs(target, messages, max_tokens, tool_choice, temperature, tools,
                                          response_format, cache_affinity, stream)
        # Official direct OpenAI Chat uses the current completion-token carrier:
        # provider-wide; model names are not capability authority across routes.
        direct_openai = provider == "openai"
        token_limit_key = "max_completion_tokens" if direct_openai else "max_tokens"
        if not target.get("supports_openrouter_extensions"):
            prepared_tools = [
                {k: v for k, v in tool.items() if k != "cache_control"}
                for tool in self._sanitize_chat_completion_tools(tools)
            ]
            # A generic compatible server gets the canonical cache hints (unknown is not
            # no); one that refuses a hint is repaired and remembered by request-wire
            # recovery on that exact route. The other direct lanes keep their strip.
            compatible = provider == "openai-compatible"
            clean_messages = self._strip_openrouter_roundtrip_metadata(
                self._copy_messages_with_cache_policy(
                    messages,
                    allow_message_cache_control=compatible,
                    flatten_tool_content_blocks=True,
                    # DeepSeek accepts content arrays only on user turns.
                    flatten_non_user_content_blocks=provider == "deepseek",
                    allow_cache_ttl=compatible,
                ),
                keep_reasoning_content=provider == "minimax" or bool(target.get("requires_reasoning_echo")),
                keep_reasoning_details=bool(target.get("reasoning_split")),
            )
            if target.get("requires_reasoning_echo"):
                # A reasoning-echo route (DeepSeek) REQUIRES every assistant
                # turn's ``reasoning_content`` on tool-bearing requests (v4-pro
                # 400s otherwise; probed 2026-09-01). Foreign or non-string
                # values become the explicit empty string the gate accepts —
                # the honest value for reasoning that does not exist. Harmless
                # without tools (the API ignores the field).
                for _msg in clean_messages:
                    if isinstance(_msg, dict) and _msg.get("role") == "assistant":
                        if not isinstance(_msg.get("reasoning_content"), str):
                            _msg["reasoning_content"] = ""
            kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "messages": clean_messages,
                token_limit_key: max_tokens,
            }
            if target.get("reasoning_split"):
                # Split thinking from final content at the provider boundary.
                # The SDK forwards this extension through extra_body, not a keyword.
                kwargs.setdefault("extra_body", {})["reasoning_split"] = True
            if stream:
                kwargs.update(stream=True, stream_options={"include_usage": True})
            if direct_openai or compatible:
                cache_identity = self._prompt_cache_identity(
                    str(target.get("usage_model") or resolved_model),
                    clean_messages,
                ) or self._explicit_cache_affinity_identity(
                    str(target.get("usage_model") or resolved_model), cache_affinity,
                )
                if cache_identity:
                    # OpenAI's named affinity key keeps requests sharing the
                    # stable governance prefix on the same cache bucket; a
                    # request without a leading system (a Light prompt) uses
                    # its caller-declared affinity instead.
                    kwargs["prompt_cache_key"] = cache_identity
            requested_effort = normalize_reasoning_effort(reasoning_effort)
            if provider not in {"deepseek", "minimax", "zai"}:
                # Every Chat Completions route carries the owner's effort in the
                # standard field: direct OpenAI, Cloud.ru and any
                # OpenAI-compatible server alike. No provider or model list
                # decides it — unknown is not no — so a route that refuses the
                # field or a value is adapted by request-wire recovery on that
                # exact route's own rejection, requested and sent kept apart.
                # DeepSeek, MiniMax and Z.ai keep their sourced projections below.
                kwargs["reasoning_effort"] = requested_effort
            elif provider == "minimax":
                # Exact Flash only; omitting none would serve max. Reuse the
                # requested/applied disclosure, including clearing a stale note.
                applied = normalize_minimax_reasoning_effort(resolved_model, requested_effort)
                kwargs["reasoning_effort"] = applied
                _stage_effort_clamp_disclosure(requested_effort, applied, resolved_model)
            elif provider == "deepseek":
                # Same carriage, projected onto DeepSeek's wire dialect
                # (low/high/max; thinking is switched off by a toggle, not an
                # effort value). Thinking mode accepts only tool_choice
                # auto/none (probed 2026-09-03: required and named 400 on both
                # v4 models), so a forced tool call is served with thinking
                # disabled. Any tier change is disclosed on usage as
                # ``reasoning_effort_clamped``.
                forced_tool = bool(prepared_tools) and tool_choice not in (None, "", "auto", "none")
                applied = "none" if forced_tool else normalize_deepseek_reasoning_effort(requested_effort)
                if applied == "none":
                    kwargs.setdefault("extra_body", {})["thinking"] = {"type": "disabled"}
                else:
                    kwargs["reasoning_effort"] = applied
                _stage_effort_clamp_disclosure(
                    requested_effort, applied, resolved_model,
                    reason="provider_forced_tool_choice" if forced_tool else "provider_wire_mapping",
                )
            elif provider == "zai":
                # Same carriage family, Z.ai's OWN projection table (NOT
                # DeepSeek's: medium does not exist at Z.ai and xhigh maps to
                # max, not high). GLM reasoning cannot be disabled — the
                # DeepSeek ``thinking={"type":"disabled"}`` arm answers
                # 400 code 1210 ("please use low, high or max") on PAYG — and
                # forced tool_choice WORKS with thinking enabled (measured
                # 2026-09-21), so there is no forced-tool exception either.
                # An absent parameter is served at MAX: dropping the tier
                # silently billed every call at max. Any tier change is
                # disclosed on usage as ``reasoning_effort_clamped``.
                applied = normalize_zai_reasoning_effort(requested_effort)
                kwargs["reasoning_effort"] = applied
                _stage_effort_clamp_disclosure(requested_effort, applied, resolved_model)
            if temperature is not None:
                kwargs["temperature"] = temperature
            if response_format:
                kwargs["response_format"] = dict(response_format)
            if prepared_tools:
                kwargs["tools"] = prepared_tools
                kwargs["tool_choice"] = tool_choice
            if headers := go_chat_headers(target):
                kwargs["extra_headers"] = headers
            if bypass_response_cache and provider == "openai-compatible":
                # Must ride in extra_body: the OpenAI SDK rejects unknown top-level
                # kwargs with TypeError, so a raw `cache=` argument never reaches
                # the wire.
                _eb = kwargs.setdefault("extra_body", {})
                if isinstance(_eb, dict):
                    _eb["cache"] = {"no-cache": True}
            apply_processing_preference(target, kwargs)
            return kwargs

        # OpenRouter cannot echo direct-provider ``reasoning_content`` (DeepSeek
        # and MiniMax), even when empty: upstreams reject it, and malformed values
        # would also pin same-model failover below. Only change the physical copy.
        messages = [dict(m) if isinstance(m, dict) else m for m in messages]
        for message in messages:
            if not isinstance(message, dict):
                continue
            message.pop("reasoning_content", None)
            flat, details = message.get("reasoning"), message.get("reasoning_details")
            # Drop only a complete exact text duplicate. All ordered entries must
            # qualify; signed details and unknown metadata remain byte-identical.
            if (message.get("role") == "assistant"
                    and isinstance(flat, str) and flat and isinstance(details, list) and details
                    and all(isinstance(entry, dict) and entry.get("type") == "reasoning.text"
                            and isinstance(entry.get("text"), str) for entry in details)
                    and "".join(entry["text"] for entry in details) == flat):
                message.pop("reasoning")
        effort = normalize_reasoning_effort(reasoning_effort)
        raw_return_reasoning = runtime_setting("OUROBOROS_RETURN_REASONING")
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

        routing = openrouter_chat_routing(target, messages)
        if routing:
            extra_body["provider"] = routing

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
        if stream:
            kwargs.update(stream=True, stream_options={"include_usage": True})
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
            for tool in prepared_tools:  # each "function" is the sanitizer's own copy
                # The Responses API serving OpenAI models here tries strict mode for a
                # tool WITHOUT `strict`, requiring every property; a set value stays.
                tool["function"].setdefault("strict", False)
            if server_web_tool:
                prepared_tools.append(server_web_tool)
            # Tool cache markers are placed once, at the send-time payload finalizer
            # (`_normalize_payload_cache_ttl`) — it is the only point that sees tools,
            # system and messages together and can order their TTLs.
            kwargs["tools"] = prepared_tools
            # OpenRouter's implicit default keeps neutral intent out of its parameter filter.
            if tool_choice != "auto":
                kwargs["tool_choice"] = tool_choice

        # With require_parameters, unsupported params cause OpenRouter 404s.
        # Unknown capabilities mean no stripping.
        if skip_capability_fetch:
            # "Skip" means skip the NETWORK fetch (no_proxy fork-safety), not
            # ignore an already-warm capability cache. The cache is per-process:
            # a process that already ran its one-shot /models fetch keeps
            # stripping unsupported params proactively instead of paying a
            # reactive 404 + retry on every reviewer call, while a freshly
            # spawned worker starts cold and strips nothing until its own fetch.
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
        apply_processing_preference(target, kwargs)
        if allow_message_cache and cache_model.startswith("anthropic/"):
            self._cache_declared_system_context(messages, kwargs)
        return kwargs

    def _responses_kwargs(self, target: Dict[str, Any], messages: List[Dict[str, Any]], max_tokens: int,
                          tool_choice: Any, temperature: Optional[float], tools: Optional[List[Dict[str, Any]]],
                          response_format: Optional[Dict[str, Any]], cache_affinity: str, stream: bool) -> Dict[str, Any]:
        """One direct OpenAI Responses body: the same sanitized tools, effort, affinity key and processing as Chat."""
        model = str(target.get("usage_model") or target.get("resolved_model") or "")
        kwargs = responses_payload(
            target, messages, effort=target["requested_reasoning_effort"], max_tokens=max_tokens,
            tool_choice=tool_choice, temperature=temperature, response_format=response_format, stream=stream,
            tools=[{k: v for k, v in tool.items() if k != "cache_control"} for tool in self._sanitize_chat_completion_tools(tools)],
            cache_identity=self._prompt_cache_identity(model, messages) or self._explicit_cache_affinity_identity(model, cache_affinity))
        apply_processing_preference(target, kwargs)
        return kwargs

    def _normalize_remote_response(
        self,
        resp_dict: Dict[str, Any],
        target: Dict[str, Any],
        skip_cost_fetch: bool = False,
        prompt_cache_ttl: Optional[str] = None,
        wire_completion: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Normalize a reply without generation lookup; skip_cost_fetch also skips tariffs.

        A Responses reply is read through ``openai_responses.completion_view`` and leaves with its exact items
        and terminal facts attached (``attach_reply_facts``); every other step is the one Chat takes."""
        responses_facts = None
        cost, cost_invalid, cost_basis = response_reported_price(
            resp_dict, dict(resp_dict.get("usage") or {}), getattr(wire_completion, "total_cost_usd", None))
        if responses_route(target):
            resp_dict, responses_facts = completion_view(resp_dict)
        usage = dict(resp_dict.get("usage") or {})
        for host_key in REPLY_FACT_KEYS:  # host-owned terminal facts: a provider usage extension never sets them
            usage.pop(host_key, None)
        if is_go_chat_route(target.get("provider"), target.get("base_url")):
            normalize_go_price(usage)
        else:
            # What the reply consumed (a BYOK reply: gateway charge + provider-billed share).
            usage["cost"] = cost
            usage.pop("cost_estimated", None)  # Only the host's estimator below sets this flag.
            usage.pop("cost_invalid", None)  # Host fact, never a provider usage extension.
            if cost_invalid:
                usage["cost_invalid"] = True
            if cost_basis:
                usage["cost_basis"] = cost_basis
        usage.pop("processing", None)  # The adapter constructs this receipt, including on compatible endpoints.
        usage.pop("service_tier", None)
        if "service_tier" in resp_dict:
            usage["service_tier"] = resp_dict["service_tier"]
        attach_processing_receipt(target, usage)
        if isinstance(resp_dict.get("_stream_receipt"), dict):
            usage["stream_receipt"] = dict(resp_dict["_stream_receipt"])
        if isinstance(usage, dict):
            # These keys are host-owned projections of designated outer fields;
            # provider usage extensions must not spoof their provenance.
            usage.pop("response_finish_reason", None)
            usage.pop("response_provider", None)
            usage.pop("reasoning_pin", None)
            usage.pop("reasoning_effort_clamped", None)
            usage.pop("provider_error", None)
            usage.pop("wire_layout", None)
            usage.pop(REPORTED_TOKENS_KEY, None)
            # The projected wire shape of THIS call's candidate (per-call target, never a
            # thread-local): the request blob keeps the canonical messages, the sealed
            # physical candidate keeps the exact wire, this fact says which one was sent.
            if isinstance(target.get("wire_layout"), dict):
                usage["wire_layout"] = dict(target["wire_layout"])
        # An HTTP-200 that carried a provider body-error (OpenRouter passes
        # 429/5xx through the body) reaches here only when a same-model reroute
        # was unavailable or also errored. Surface it as a typed marker so the
        # caller classifies it as a real rate_limit/provider_transient instead of
        # a blank finish_reason=null "incomplete response".
        _body_err = self._provider_body_error(resp_dict)
        if _body_err:
            usage["provider_error"] = body_error_facts(_body_err, self._is_transient_body_error(_body_err))
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
        # DeepSeek is the inverse class: its documented tool contract REQUIRES the
        # previous turns' ``reasoning_content`` back on every tools-bearing request
        # (v4-pro enforces with a 400), so that lane KEEPS the field on the canonical
        # assistant message — the same-family-continuity treatment ``reasoning_details``
        # already gets. Cross-family sends strip it (sanitize_reasoning_on_model_switch
        # + the outbound scrubber), and the deepseek outbound build replays it.
        provider = str(target.get("provider") or "")
        if (provider == "openai-compatible" and isinstance(msg.get("content"), str)
                and not any(msg.get(key) for key in ("reasoning_content", "reasoning", "reasoning_details"))):
            # A local server (llama.cpp without a reasoning parser, vLLM) can return the model's leading reasoning, or
            # its lone closing ``</think>``, inside the answer; a reply with its own reasoning field is not read. The
            # built-in lane's reader splits it off (``llm_local._split_leading_reasoning``); the reasoning stays beside
            # the answer as the portable ``reasoning`` the outbound scrubber drops. An answer left empty takes the
            # empty-response recovery. A reply with tool calls keeps its history (llama.cpp #24890) and gains its
            # display projection (``llm_local.stamp_display``).
            from ouroboros.llm_local import _split_leading_reasoning, stamp_display

            if msg.get("tool_calls"):
                stamp_display(msg)
            elif (split := _split_leading_reasoning(msg["content"]))[0] != msg["content"]:
                msg["content"] = split[0]
                if split[1]:
                    msg["reasoning"] = split[1]
        # MiniMax's two continuation carriers stay verbatim, including unknown
        # shapes. Retention does not make an opaque value displayable or portable.
        if provider not in {"deepseek", "minimax"}:
            msg.pop("reasoning_content", None)
        elif provider == "deepseek" and not isinstance(msg.get("reasoning_content", ""), str):
            # The SDK surfaces server extras verbatim (same hazard the
            # refusal/annotations pops above guard): a null here would live on
            # the canonical assistant turn forever and the direct lane has no
            # message-level 400 recovery. Only strings enter the transcript.
            msg.pop("reasoning_content", None)

        # The billing-shaped counters hold a reported count or nothing: a malformed one (negative, fractional,
        # not a number) is neither lifted nor left for a later fold or tariff to read as a number.
        malformed = malformed_token_counts(usage)
        # The ledger's reading of the reply as reported (``cache_read_count``): the details block wins, DeepSeek's
        # top-level prompt_cache_hit_tokens mirror is the fallback, and a reported zero is a measurement.
        cache_read = cache_read_count(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            if usage.get(key) is not None:
                usage[key] = _reported_token_count(usage, key)
        # A positive count is lifted to the top level. A zero is written there only when the reply's other names
        # would read as something else in the event (``event_token_counts``, whose own order puts the top-level
        # names first); reported nowhere else, it stays where the reply put it.
        if cache_read or (cache_read == 0 and event_token_counts(usage)["cached_tokens"] != 0):
            usage["cached_tokens"] = cache_read
        # LM Studio MLX exposes prefix-cache hits only in stderr/logs, not
        # OpenAI-compatible usage; cached_tokens=0 is therefore expected.

        write_split = self._cache_write_split(usage)
        written, invalid_write = cache_write_total(usage, write_split)
        usage.pop("cache_write_tokens_invalid", None)
        if invalid_write:
            usage["cache_write_tokens_invalid"] = True
            usage["cache_write_tokens"] = written
        if write_split:
            usage["cache_write_tokens_by_ttl"] = write_split
        # A zero reported only inside Responses details stays there. A complete pair of bins establishes a total.
        if written is not None and (responses_facts is None or written or write_split or "cache_write_tokens" in usage):
            usage["cache_write_tokens"] = written

        usage["provider"] = str(target.get("provider") or "openrouter")
        usage["resolved_model"] = str(target.get("usage_model") or target.get("resolved_model") or "")
        terminal = current_wire_candidate()  # the TTL the terminal candidate carried, after any hint repair
        prompt_cache_ttl = _applied_payload_cache_ttl(terminal.physical_payload()) if terminal is not None else prompt_cache_ttl
        if prompt_cache_ttl and not usage.get("prompt_cache_ttl"):
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        if usage["provider"] == "deepseek" and usage.get("cost") is None and not usage.get("cost_invalid"):
            from ouroboros._usage_settlement import settled_cost_facts

            usage.update(settled_cost_facts(usage))  # its attempt row's pinned amount, never a fresh price
        counts = text_token_counts(usage)  # the ledger's reading: an unreported count stays None, never 0
        if (usage.get("cost") is None and not usage.get("cost_invalid") and not usage.get("provider_reported_cost_only")
                and any(counts)):
            from ouroboros._usage_settlement import has_nontext_usage
            from ouroboros.pricing import estimate_cost_optional

            # The ledger's rules: reported audio or image usage is never priced as text, and a malformed count is
            # no observed zero. A reported bill stays.
            estimated_cost = None if has_nontext_usage(usage) or malformed else estimate_cost_optional(
                usage["resolved_model"],
                *counts,
                cache_usage=cache_pricing_usage(usage, written, write_split),
                allow_live_fetch=not skip_cost_fetch,
                provider=usage["provider"],
                **({"processing_mode": observed_processing_mode(usage["provider"], usage)}
                   if usage.get("processing") else {}),
            )
            if estimated_cost is not None:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True
        if usage.get("cost") is None:
            usage["cost"] = None
            if malformed:  # settlement's own fact: the loop's estimator stays off the malformed count as well
                usage["cost_coverage"] = "partial"
        usage["cost_final"] = bool(
            usage.get("cost") is not None and not usage.get("cost_estimated")
        )
        # Merge the tier note staged at build (DeepSeek/MiniMax/Z.ai projections); normal
        # dispatch adaptation is disclosed through usage.request_wire.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note
        # Same disclosure norm for a ≤4-cap cache-marker reduction (v6.77.0): never silent.
        _cache_note = self._pop_thread_disclosure("_cache_breakpoint_tls")
        if _cache_note:
            usage["prompt_cache_breakpoints_reduced"] = _cache_note
        # Why same-model provider failover was withheld on THIS call (issue #468):
        # typed, bounded, and read off the terminal sent candidate — never a guess.
        from ouroboros.reasoning_artifacts import pop_reasoning_pin_note

        _pin_note = pop_reasoning_pin_note()
        if _pin_note:
            usage["reasoning_pin"] = _pin_note
        from ouroboros.openai_chat_dispatch import normalize_direct_openai_completion

        if responses_facts is not None:
            msg, usage = attach_reply_facts(msg, usage, responses_facts, target)
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



def openrouter_web_search_server_tool(
    *,
    api_key: str,
    model: str,
    query: str,
    search_context_size: str,
    accounting_scope: Optional[UsageScope] = None,
    timeout: Optional[float] = None,
    processing_preference: str | None = None,
    target: Optional[Dict[str, Any]] = None,
    _recovery: Any,
) -> Any:
    """Run OpenRouter's provider-owned web_search server tool.

    ``target`` is the search leg's resolved connection (``llm_routing.resolve_target``): the send carries its
    endpoint, credential, headers, redirect rule, billing identity (``usage_model``) and provider routing, merged
    as a chat send merges it (``_resolve_or_provider``), so a named connection's restrictions hold and it is
    priced as that connection. Without one: the built-in public connection with ``api_key`` and the app identity.
    The tier is projected as on any send; the server tool's per-use charge leaves the reservation unknown
    (``llm_routing.carries_add_on_charge``)."""

    from ouroboros.net_transport import web_search_openai_client
    from ouroboros.model_slots import resolve_processing_preference
    from ouroboros.usage_accounting import current_usage_scope
    from dataclasses import replace

    from ouroboros.llm_routing import OPENROUTER_BASE_URL

    target = dict(target) if target is not None else {
        "provider": "openrouter", "usage_model": model, "resolved_model": model,
        "base_url": OPENROUTER_BASE_URL, "api_key": api_key, "default_headers": dict(OPENROUTER_APP_HEADERS)}
    target["processing_preference"] = resolve_processing_preference("websearch", override=processing_preference)

    client = web_search_openai_client(
        api_key=target["api_key"],
        base_url=target["base_url"],
        timeout=timeout,
        default_headers=dict(target.get("default_headers") or {}),
        follow_redirects=target.get("follow_redirects") is not False,
    )
    payload = dict(
        model=str(target.get("resolved_model") or model),
        messages=[{"role": "user", "content": query}],
        tools=[{
            "type": "openrouter:web_search",
            "parameters": {
                "search_context_size": search_context_size,
                "max_total_results": 10,
            },
        }],
    )
    routing = _resolve_or_provider(target.get("provider_routing"))
    if routing:
        payload["extra_body"] = {"provider": routing}
    apply_processing_preference(target, payload)
    if timeout is not None:
        payload["timeout"] = float(timeout)  # the leg's own bound: each send narrows it, never replaces it (#572)
    scope = replace(accounting_scope or current_usage_scope() or UsageScope(), source="web_search.openrouter")
    with usage_scope(scope):
        return _recovery(client.chat.completions.create, payload, target, search_server_tool=True)
