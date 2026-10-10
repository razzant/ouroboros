"""LLM client for OpenRouter, direct providers, and optional local inference."""

from __future__ import annotations

import asyncio
import copy  # noqa: F401  (prior import surface)
import hashlib  # noqa: F401  (prior import surface)
import inspect  # noqa: F401  (prior import surface)
import json
import logging
import re
import threading  # noqa: F401  (prior import surface)
import time  # noqa: F401  (prior import surface)
from typing import Any, Dict, List, Optional, Set, Tuple

from ouroboros.model_wait import dispatch_deadline_remaining_sec, model_waitable
from ouroboros.deadline_utils import caller_deadline_scoped

from ouroboros.anthropic_native_custody import (  # noqa: F401  (prior import surface)
    anthropic_replay_scoped,
    custody_private_key,
    is_replayed_native_content,
    mark_replayed_receipts_consumed,
    native_content_for_replay,
    retain_native_assistant_content,
    scrub_native_custody,
)
from ouroboros.context_budget import (  # noqa: F401
    CONTEXT_OVERFLOW_CODES,
    context_overflow_message,
)
from ouroboros.llm_anthropic import (

    anthropic_web_search_server_tool as _anthropic_web_search_server_tool,
    _AnthropicLaneMixin,  # noqa: F401
)
from ouroboros.llm_attempt import (
    _attempt_request,  # noqa: F401 -- historical public import surface
    PROVIDER_POLICY_REFUSAL,  # noqa: F401
    ProviderPolicyRefusal,  # noqa: F401
    _applied_payload_cache_ttl,  # noqa: F401
    _CACHE_TTL_SECONDS,  # noqa: F401
    _canonical_candidate_bytes,  # noqa: F401
    _candidate_before_dispatch,  # noqa: F401 -- historical public import surface
    _execute_candidate_async,  # noqa: F401
    _execute_candidate,  # noqa: F401 -- historical public import surface
    _physical_candidate,  # noqa: F401 -- historical public import surface
    _finalized_physical_candidate,  # noqa: F401
    _is_provider_policy_refusal,  # noqa: F401
    _is_structured_context_overflow_body,  # noqa: F401
    _is_structured_context_overflow_exception,  # noqa: F401
    _PayloadCachePolicyMixin,  # noqa: F401
    _route_normalizes_cache_breakpoints,  # noqa: F401
    _structured_error_values,  # noqa: F401
    _VALID_CACHE_TTLS,  # noqa: F401
    cache_ttl_seconds,  # noqa: F401
    supports_message_cache_control,  # noqa: F401
    apply_processing_preference,  # noqa: F401 -- historical public import surface
    processing_contract_headers,  # noqa: F401 -- historical public import surface
)
from ouroboros.llm_capability_policy import (
    _CapabilityPolicyMixin,  # noqa: F401
    _MANDATORY_VALUE_MARKERS,  # noqa: F401
    _OPTIONAL_DROPPABLE_PARAMS,  # noqa: F401
    _OPTIONAL_SAMPLING_PARAMS,  # noqa: F401
    normalize_reasoning_effort,  # noqa: F401
)
from ouroboros.llm_fallback import (
    _RecoveryLadderMixin,  # noqa: F401
)
from ouroboros.llm_gigachat import (
    _GigaChatLaneMixin,  # noqa: F401
)
from ouroboros.llm_local import (
    _compact_local_text,  # noqa: F401
    _compact_markdown_sections,  # noqa: F401
    _estimate_message_chars,  # noqa: F401
    _LOCAL_COMPACTION_MODES,  # noqa: F401
    _LocalLaneMixin,  # noqa: F401
    _split_markdown_sections,  # noqa: F401
    LocalContextTooLargeError,  # noqa: F401
)
from ouroboros.llm_messages import (
    _MessageShapingMixin,  # noqa: F401
)
from ouroboros.llm_openai_compatible import (

    openrouter_web_search_server_tool as _openrouter_web_search_server_tool,
    _bounded_response_metadata_label,  # noqa: F401
    _FALSE_LIKE_ENV_VALUES,  # noqa: F401
    _OpenAICompatibleLaneMixin,  # noqa: F401
    _RESPONSE_METADATA_LABEL_MAX_CHARS,  # noqa: F401
)
from ouroboros.llm_pricing import (
    add_usage,  # noqa: F401
    fetch_cloudru_pricing,  # noqa: F401
    fetch_openrouter_pricing,  # noqa: F401
)
from ouroboros.llm_routing import (
    _OR_PROVIDER_PRESETS,  # noqa: F401
    _ProviderRoutingMixin,  # noqa: F401
    _resolve_or_provider,  # noqa: F401
)
from ouroboros.provider_models import (  # noqa: F401  (prior import surface)
    DEEPSEEK_BASE_URL,
    OPENROUTER_DEFAULTS,
    PROVIDER_PREFIXES,
    normalize_anthropic_model_id,
    normalize_deepseek_reasoning_effort,
    normalize_model_identity,
    resolve_minimax_base_url,
)
from ouroboros.request_wire_recovery import (
    finalize_wire_response,  # noqa: F401
    note_provider_metadata_drop_fields,  # noqa: F401
    note_wire_send_failed,  # noqa: F401
    note_wire_send_succeeded,  # noqa: F401
    plan_next_wire_retry,  # noqa: F401
    prepare_wire_payload_for_send,  # noqa: F401
    request_wire_scoped,
)
from ouroboros.transport_custody import is_loopback_base_url  # noqa: F401
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS  # noqa: F401 -- historical public import surface
from ouroboros.usage_accounting import (
    AttemptRequest,  # noqa: F401
    PhysicalAttemptCapture,  # noqa: F401
    PhysicalAttemptPreconditionFailed,  # noqa: F401
    PhysicalAttemptPreparationFailed,  # noqa: F401
    UsageAccountingError,  # noqa: F401
    UsageScope,
    capture_attempt_ids,
    current_physical_attempt_context,  # noqa: F401
    current_physical_attempt_predicate,  # noqa: F401
    current_usage_scope,  # noqa: F401
    execute_physical_attempt,  # noqa: F401
    execute_physical_attempt_async,  # noqa: F401
    last_physical_attempt_capture,  # noqa: F401
    usage_scope,  # noqa: F401 -- historical public import surface
    )
from ouroboros.utils import in_worker_process, sanitize_tool_result_for_log  # noqa: F401
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = OPENROUTER_DEFAULTS["light"]


class LLMClient(
    _PayloadCachePolicyMixin,
    _CapabilityPolicyMixin,
    _ProviderRoutingMixin,
    _MessageShapingMixin,
    _RecoveryLadderMixin,
    _AnthropicLaneMixin,
    _GigaChatLaneMixin,
    _LocalLaneMixin,
    _OpenAICompatibleLaneMixin,
):
    """LLM API wrapper. Routes calls to OpenRouter or a local llama-cpp-python server."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key_override = api_key
        self._api_key = api_key or runtime_setting("OPENROUTER_API_KEY", "")
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

    @caller_deadline_scoped
    @model_waitable
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
        model_role: str = "",
        model_poll_control: Any = None,
        model_operation_observer: Any = None,
        model_account_override: str | None = None,
        model_turn_state: Any = None,
        default_temperature: Optional[float] = None,
        stream: bool = False,
        caller_deadline_ts: Optional[float] = None,
        caller_execution_deadline: Optional[float] = None,
        wait_for_resources: bool = True,
        processing_preference: str | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call returning (message, usage); no_proxy avoids macOS fork proxy crashes.

        ``response_format`` (e.g. ``{"type": "json_object"}``) is optional request
        intent on the OpenAI-compatible/OpenRouter lanes: local, Anthropic-native,
        and GigaChat routes ignore it, and a provider rejection strips it via the
        optional-parameter retry — callers must keep a text-parse fallback.

        ``default_temperature`` is a host hint, unlike explicit ``temperature``.
        Resolve it on this invocation's effective route after any model wait:
        raw model operations defer to provider defaults; explicit values win.

        ``model_turn_state`` is the caller's optional active-turn transport slot
        (``llm_claudexor.ModelTurnState``); this seam is where a dispatch that
        leaves that transport ends the turn. ``wait_for_resources=False`` returns
        a confirmed quota/auth refusal to this caller without entering resource
        waiting; task overrides, controls and dispatched-operation custody remain."""
        from ouroboros.llm_claudexor import turn_state_for_route

        messages = self._normalize_system_message_placement(messages)
        with capture_attempt_ids() as attempt_ids:
            if use_local:
                turn_state_for_route(model_turn_state, "local")
                local_kwargs = {"timeout": timeout}
                if reasoning_effort is not None:
                    local_kwargs["reasoning_effort"] = reasoning_effort
                if processing_preference:
                    local_kwargs["processing_preference"] = processing_preference
                message, usage = self._chat_local(
                    messages, tools, max_tokens, tool_choice, **local_kwargs,
                )
            else:
                # Central worker policy: remote calls from worker processes avoid
                # system proxy lookup without every caller remembering a flag.
                no_proxy = no_proxy or in_worker_process()
                target = {**self._resolve_remote_target(model), "processing_preference": processing_preference}
                if temperature is None and target.get("provider") != "claudexor":
                    temperature = default_temperature
                message, usage = self._chat_remote(
                    target, messages, tools, reasoning_effort, max_tokens, tool_choice, temperature,
                    no_proxy=no_proxy,
                    timeout=timeout,
                    allow_server_web_search=allow_server_web_search,
                    response_format=response_format,
                    cache_affinity=cache_affinity,
                    bypass_response_cache=bypass_response_cache,
                    model_role=model_role,
                    model_poll_control=model_poll_control,
                    model_operation_observer=model_operation_observer,
                    model_account_override=model_account_override,
                    model_turn_state=turn_state_for_route(model_turn_state, target.get("provider")),
                    **({"stream": True} if stream else {}),
                )
            usage["ledger_attempt_ids"] = list(attempt_ids)
            return message, usage

    @request_wire_scoped
    @caller_deadline_scoped
    @model_waitable
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
        model_role: str = "",
        model_poll_control: Any = None,
        model_operation_observer: Any = None,
        model_account_override: str | None = None,
        model_turn_state: Any = None,
        use_local: bool = False,
        default_temperature: Optional[float] = None,
        stream: bool = False,
        caller_deadline_ts: Optional[float] = None,
        caller_execution_deadline: Optional[float] = None,
        wait_for_resources: bool = True,
        processing_preference: str | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async remote chat; no_proxy keeps forked macOS workers off OS proxy APIs.

        Host temperature hints, resource waiting and the active-turn transport
        slot follow ``chat``'s effective-route contract."""
        from ouroboros.llm_claudexor import turn_state_for_route

        messages = self._normalize_system_message_placement(messages)
        no_proxy = no_proxy or in_worker_process()
        if use_local:
            turn_state_for_route(model_turn_state, "local")
            from ouroboros.usage_accounting import adopt_physical_attempt_capture, last_physical_attempt_capture

            def local_call():
                adopt_physical_attempt_capture(None)
                local_kwargs = {"timeout": timeout}
                if reasoning_effort is not None:
                    local_kwargs["reasoning_effort"] = reasoning_effort
                if processing_preference:
                    local_kwargs["processing_preference"] = processing_preference
                result = self._chat_local(messages, tools, max_tokens, tool_choice, **local_kwargs)
                return result, last_physical_attempt_capture()

            with capture_attempt_ids() as attempt_ids:
                try:
                    result, capture = await asyncio.to_thread(local_call)
                except BaseException as exc:
                    adopt_physical_attempt_capture(getattr(exc, "physical_attempt_capture", None))
                    raise
                adopt_physical_attempt_capture(capture)
            result[1]["ledger_attempt_ids"] = list(attempt_ids)
            return result
        target = {**self._resolve_remote_target(model), "processing_preference": processing_preference}
        if temperature is None and target.get("provider") != "claudexor":
            temperature = default_temperature
        carried_turn_state = turn_state_for_route(model_turn_state, target.get("provider"))
        if target.get("provider") == "claudexor":
            from ouroboros.llm_claudexor import chat_claudexor_async

            with capture_attempt_ids() as attempt_ids:
                result = await chat_claudexor_async(
                    target, messages, tools, reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens, tool_choice=tool_choice, temperature=temperature,
                    timeout=timeout, allow_server_web_search=allow_server_web_search,
                    cache_affinity=cache_affinity, model_role=model_role,
                    model_poll_control=model_poll_control,
                    model_operation_observer=model_operation_observer,
                    model_account_override=model_account_override,
                    model_turn_state=carried_turn_state,
                )
            result[1]["ledger_attempt_ids"] = list(attempt_ids)
            return result
        if target.get("provider") == "anthropic":
            with capture_attempt_ids() as attempt_ids:
                result = await self._chat_anthropic_async(
                    target, messages, tools, reasoning_effort,
                    max_tokens, tool_choice, temperature, no_proxy, timeout,
                    **({"stream": True} if stream else {}),
                )
            result[1]["ledger_attempt_ids"] = list(attempt_ids)
            return result
        if target.get("provider") == "gigachat":
            if tools:
                raise ValueError("chat_async does not support GigaChat tool calls")
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
                    **({"stream": True} if stream else {}),
                )
                if dispatch_deadline_remaining_sec() is not None:
                    kwargs["timeout"] = self._no_proxy_timeout(timeout)
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
                    from ouroboros._usage_wait import aclose_after_model_send
                    await aclose_after_model_send(_http_client)
                except Exception:
                    pass
        client = self._get_async_remote_client(target)
        kwargs = self._build_remote_kwargs(
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
            allow_server_web_search=allow_server_web_search,
            cache_affinity=cache_affinity,
            **({"stream": True} if stream else {}),
        )
        if timeout and timeout > 0:
            # Cached clients are built without a timeout; honor the caller's
            # per-request timeout instead of silently using the SDK default.
            kwargs["timeout"] = float(timeout)
        elif dispatch_deadline_remaining_sec() is not None:
            kwargs["timeout"] = getattr(client, "timeout", None)
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

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = DEFAULT_LIGHT_MODEL,
        max_tokens: int = 32768,
        reasoning_effort: str = "medium",
        timeout: float = 90.0,
        *,
        model_role: str = "vision",
        use_local: bool = False,
        model_poll_control: Any = None,
        model_operation_observer: Any = None,
        model_account_override: str | None = None,
        processing_preference: str | None = None,
        purpose: str = "vlm",
    ) -> Tuple[str, Dict[str, Any]]:
        """Run a lightweight vision query; image dicts use url or base64+mime.

        ``model`` is named explicitly, so the one image policy sends its pixels
        (``purpose`` "vlm" for a VLM tool, "caption" for a send-time caption);
        it never captions here, so a caption cannot recurse into another."""
        from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send
        from ouroboros.vision_image_limits import query_image_messages
        from ouroboros.model_wait import current_model_wait
        from contextlib import nullcontext

        canonical = query_image_messages(prompt, images)
        sent = {}

        def prepare(values):
            from ouroboros.vision_routing import _has_image, _image_digest, _is_image

            messages = prepare_messages_for_send(canonical, routing=VisionRoutingContext(
                values["model"], self, {}, use_local=values.get("use_local", False),
                model_role=values.get("model_role", model_role),
                model_account_override=values.get("model_account_override")), purpose=purpose)
            parts = [part for message in messages for part in message.get("content", [])]
            sent.update(model=values["model"], digests=[_image_digest(part) for part in parts if _is_image(part)],
                        note=" ".join(part["text"] for part in parts[1:] if part.get("type") == "text"))
            if _has_image(canonical) and not sent["digests"]:
                error = ValueError(f"VLM_NO_IMAGE_PIXELS: {values['model']}: {sent['note']}")
                error.vision_query_receipt = dict(sent)
                raise error
            return {**values, "messages": messages}

        values = prepare(dict(
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            no_proxy=True,
            timeout=timeout,
            use_local=use_local,
            model_role=model_role,
            model_poll_control=model_poll_control,
            model_operation_observer=model_operation_observer,
            model_account_override=model_account_override,
            processing_preference=processing_preference,
        ))
        waiter = current_model_wait()
        with waiter.register_reprepare(model_role, prepare) if waiter else nullcontext():
            try:
                response_msg, usage = self.chat(**values)
            except Exception as exc:
                exc.vision_query_receipt = dict(sent)
                raise
        usage = {**usage, "_vision_query_receipt": dict(sent)}
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return runtime_setting("OUROBOROS_MODEL", OPENROUTER_DEFAULTS["main"])

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = self.default_model()
        light = runtime_setting("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if light and light != main:
            models.append(light)
        return models


def openrouter_web_search_server_tool(

    *, api_key: str, model: str, query: str, search_context_size: str,
    accounting_scope: Optional[UsageScope] = None, timeout: Optional[float] = None,
    processing_preference: str | None = None,
) -> Any:
    """Keep native search tools on the shared physical-send recovery driver."""
    return _openrouter_web_search_server_tool(
        api_key=api_key, model=model, query=query, search_context_size=search_context_size,
        accounting_scope=accounting_scope, timeout=timeout, processing_preference=processing_preference,
        _recovery=LLMClient()._create_chat_completion_with_retries,
    )


def anthropic_web_search_server_tool(
    *, api_key: str, model: str, query: str,
    accounting_scope: Optional[UsageScope] = None, timeout: Optional[float] = None,
    processing_preference: str | None = None,
) -> Any:
    """Compose native Messages search without converting its provider-owned tool."""
    return _anthropic_web_search_server_tool(
        api_key=api_key, model=model, query=query, accounting_scope=accounting_scope,
        timeout=timeout, processing_preference=processing_preference,
        _recovery=LLMClient()._create_chat_completion_with_retries,
    )
