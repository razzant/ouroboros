"""Web search tool — OpenAI Responses API with LLM-first overridable defaults."""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ouroboros.pricing import estimate_cost_optional
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture,
    UsageAccountingError,
    UsageScope,
    capture_attempt_ids,
    current_usage_scope,
    mark_dispatched,
    mark_unresolved,
    release_attempt,
    reserve_attempt,
    settle_attempt,
)
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)

from ouroboros.search_routes import SEARCH_MODELS, resolve_web_search_route

DEFAULT_SEARCH_MODEL = SEARCH_MODELS["openai"]
DEFAULT_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_REASONING_EFFORT = "high"


def _wrapped_provider_error(provider: str, exc: Exception) -> RuntimeError:
    """Sanitize a provider error without dropping its physical-attempt fact."""
    detail = sanitize_tool_result_for_log(str(exc))[:500]
    wrapped = RuntimeError(f"{provider} web search failed ({type(exc).__name__}): {detail}")
    wrapped.ledger_attempt_ids = list(getattr(exc, "ledger_attempt_ids", []) or [])
    capture = getattr(exc, "physical_attempt_capture", None)
    if isinstance(capture, PhysicalAttemptCapture):
        setattr(wrapped, "physical_attempt_capture", capture)
    return wrapped


def _provider_outcome_is_unknown(exc: Exception) -> bool:
    """Whether paid work started without a typed terminal provider outcome."""
    from ouroboros.transport_custody import outcome_unknown_on_chain

    capture = getattr(exc, "physical_attempt_capture", None)
    return bool(
        isinstance(capture, PhysicalAttemptCapture)
        and outcome_unknown_on_chain(exc)
        and capture.provider_status_code is None
    )


def _unknown_provider_outcome(backend: str) -> str:
    return json.dumps({
        "error": (
            f"{backend} web search was dispatched but its provider outcome is unknown; "
            "no retry or paid fallback was sent."
        ),
        "backend": backend,
        "reason_code": "provider_outcome_unknown",
    }, ensure_ascii=False, indent=2)


def _web_search_transport_timeout(ctx: Any) -> float:
    """One per-attempt dead-socket bound, narrowed by the owner deadline."""
    from ouroboros.config import get_websearch_timeout_sec
    from ouroboros.deadline_utils import transport_timeout_with_deadline
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    metadata = getattr(ctx, "task_metadata", {})
    deadline_at = metadata.get("deadline_at") if isinstance(metadata, dict) else None
    return transport_timeout_with_deadline(
        get_websearch_timeout_sec(),
        deadline_at=deadline_at,
        reserve_sec=effective_finalization_reserve_sec(ctx),
    )


def _web_search_deadline_exhausted(ctx: Any) -> bool:
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context
    from ouroboros.task_pacing import effective_finalization_reserve_sec
    return owner_deadline_exhausted_for_context(ctx, reserve_sec=effective_finalization_reserve_sec(ctx))


def _web_search_deadline_result() -> str:
    return json.dumps({
        "error": "web_search skipped: owner deadline leaves no dispatch window",
        "reason_code": "deadline_exhausted",
        "backend": "web_search",
    }, ensure_ascii=False, indent=2)


def _accounting_scope(ctx: ToolContext, source: str) -> UsageScope:
    bound_scope = current_usage_scope()
    if bound_scope is not None:
        return replace(bound_scope, category="web_search", source=source)

    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw_task_id = getattr(ctx, "task_id", "") or metadata.get("task_id") or ""
    task_id = str(raw_task_id) if isinstance(raw_task_id, (str, int)) else ""
    raw_root_task_id = metadata.get("root_task_id") or task_id
    root_task_id = str(raw_root_task_id) if isinstance(raw_root_task_id, (str, int)) else task_id
    raw_parent_task_id = metadata.get("parent_task_id") or ""
    raw_drive_root = (
        metadata.get("budget_drive_root") or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root", None)
    )
    drive_root = raw_drive_root if isinstance(raw_drive_root, (str, pathlib.Path)) else None
    return UsageScope(
        drive_root=drive_root,
        task_id=task_id,
        root_task_id=root_task_id,
        parent_task_id=(str(raw_parent_task_id) if isinstance(raw_parent_task_id, (str, int)) else ""),
        category="web_search",
        source=source,
    )


def _estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Direct OpenAI Responses cost is unknown without provider-reported cost."""
    pricing_model = model if "/" in str(model or "") else f"openai/{model}"
    return estimate_cost_optional(
        pricing_model, input_tokens, output_tokens, provider="openai",
    )


def _obj_to_plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _obj_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_obj_to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _obj_to_plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): _obj_to_plain(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def _extract_sources_from_response(resp_obj: Any) -> List[Dict[str, str]]:
    plain = _obj_to_plain(resp_obj)
    sources: List[Dict[str, str]] = []
    seen: set[str] = set()

    stack: List[Any] = [plain]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ntype = str(node.get("type") or "").lower()
            if "url_citation" in ntype or ("url" in node and ("title" in node or "snippet" in node)):
                url = sanitize_tool_result_for_log(str(node.get("url") or node.get("uri") or "").strip())
                if url and url not in seen:
                    seen.add(url)
                    sources.append({
                        "url": url,
                        "title": sanitize_tool_result_for_log(str(node.get("title") or node.get("name") or "").strip()),
                        "snippet": sanitize_tool_result_for_log(str(
                            node.get("snippet") or node.get("text") or node.get("content")
                            or node.get("cited_text") or node.get("description") or ""
                        ).strip()),
                    })
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    return sources


def _resolve_openai_client_settings() -> tuple[str, str | None, str, str]:
    """Return credentials only for official OpenAI Responses web search."""
    official_key = (runtime_setting("OPENAI_API_KEY", "") or "").strip()
    legacy_base_url = (runtime_setting("OPENAI_BASE_URL", "") or "").strip()

    if official_key and not legacy_base_url:
        return official_key, None, "openai", "openai"
    return "", None, "openai", "openai"


def _openrouter_model(model: str) -> str:
    route = resolve_web_search_route(backend="openrouter", model=model or None)
    legs = route["legs"] + route["unavailable"]
    if not legs:
        raise ValueError(route.get("error", "Search model unavailable"))
    return legs[0]["model"]


def _anthropic_model(model: str) -> str:
    route = resolve_web_search_route(backend="anthropic", model=model or None)
    legs = route["legs"] + route["unavailable"]
    if not legs:
        raise ValueError(route.get("error", "Search model unavailable"))
    return legs[0]["model"]


def _available_web_search_backends() -> list[str]:
    return [leg["backend"] for leg in resolve_web_search_route()["legs"]]


def _web_search_backend_pin() -> str:
    return (runtime_setting("OUROBOROS_WEBSEARCH_BACKEND") or "").strip().lower()


def _web_search_outer_timeout_sec() -> int:
    """Cover every configured paid leg absent a tighter owner deadline."""
    from ouroboros.config import get_finalization_grace_sec, get_websearch_timeout_sec

    pinned = _web_search_backend_pin()
    if pinned == "openai":
        attempts = 2
    elif pinned in {"openrouter", "anthropic"}:
        attempts = 1
    elif pinned == "ddgs":
        attempts = 0
    else:
        backends = set(_available_web_search_backends())
        attempts = 2 * ("openai_responses" in backends)
        attempts += "openrouter_server_tool" in backends
        attempts += "anthropic_server_tool" in backends
    return max(
        600,
        int(max(2, attempts) * get_websearch_timeout_sec() + get_finalization_grace_sec()),
    )


def _emit_simple_usage(
    ctx: ToolContext,
    *,
    provider: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    usage: Dict[str, Any] | None = None,
) -> None:
    if not hasattr(ctx, "pending_events"):
        return
    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        from ouroboros._usage_response import observed_processing_mode, provider_cost_value

        mode = observed_processing_mode(provider, usage or {})
        cost = provider_cost_value((usage or {}).get("cost"))
        if cost is None:
            cost = estimate_cost_optional(
                model if "/" in str(model) else f"{provider}/{model}",
                prompt_tokens,
                completion_tokens,
                provider=provider,
                **({"processing_mode": mode} if mode else {}),
            )
        ctx.pending_events.append({
            "type": "llm_usage",
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "root_task_id": str(metadata.get("root_task_id") or ""),
            "parent_task_id": str(metadata.get("parent_task_id") or ""),
            "delegation_role": str(metadata.get("delegation_role") or ""),
            "provider": provider,
            "model": model,
            "api_key_type": provider,
            "model_category": "websearch",
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "usage": usage or {},
            "cost": cost,
            "source": "web_search",
            "ts": utc_now_iso(),
            "category": "task",
            "accounting_authority": "physical_attempt_ledger",
            "ledger_attempt_ids": list((usage or {}).get("ledger_attempt_ids") or []),
        })
    except Exception:
        log.debug("Failed to emit web_search fallback cost event", exc_info=True)


def _web_search_openrouter(ctx: ToolContext, query: str, model: str = "", search_context_size: str = "",
                           processing_preference: Optional[str] = None) -> str:
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    api_key = str(runtime_setting("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    try:
        from ouroboros.llm import openrouter_web_search_server_tool

        active_model = _openrouter_model(model)
        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Searching via OpenRouter: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        with capture_attempt_ids() as attempt_ids:
            response = openrouter_web_search_server_tool(
                api_key=api_key,
                model=active_model,
                query=query,
                search_context_size=search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE,
                accounting_scope=_accounting_scope(ctx, "web_search.openrouter"),
                timeout=_web_search_transport_timeout(ctx),
                processing_preference=processing_preference,
            )
        message = response.choices[0].message if getattr(response, "choices", None) else None
        text = str(getattr(message, "content", "") or "").strip()
        from ouroboros._usage_response import usage_from_response
        from ouroboros.llm_attempt import attach_processing_receipt

        usage, _cost, _final = usage_from_response(response)
        attach_processing_receipt({"provider": "openrouter", "usage_model": active_model}, usage)
        # Guard: an exotic (non-dict) usage object must not crash the leg with a
        # successful paid answer in hand — mirror the Anthropic leg's isinstance
        # check (a `.get` on a non-dict would raise and discard the result).
        if not isinstance(usage, dict):
            usage = {}
        usage["ledger_attempt_ids"] = list(attempt_ids)
        _emit_simple_usage(
            ctx,
            provider="openrouter",
            model=active_model,
            prompt_tokens=int((usage or {}).get("prompt_tokens") or 0),
            completion_tokens=int((usage or {}).get("completion_tokens") or 0),
            usage=usage if isinstance(usage, dict) else {},
        )
        return json.dumps({
            "answer": text or "(no answer)",
            "answer_type": "summary",
            "sources": _extract_sources_from_response(response),
            "backend": "openrouter_server_tool",
            "model": active_model, "reported_model": getattr(response, "model", None), "ledger_attempt_ids": list(attempt_ids),
        }, ensure_ascii=False, indent=2)
    except UsageAccountingError:
        raise
    except Exception as exc:
        raise _wrapped_provider_error("OpenRouter", exc) from exc


def _web_search_anthropic(ctx: ToolContext, query: str, model: str = "",
                          processing_preference: Optional[str] = None) -> str:
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    api_key = str(runtime_setting("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    try:
        from ouroboros.llm import anthropic_web_search_server_tool

        active_model = _anthropic_model(model)
        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Searching via Anthropic: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        with capture_attempt_ids() as attempt_ids:
            response = anthropic_web_search_server_tool(
                api_key=api_key,
                model=active_model,
                query=query,
                accounting_scope=_accounting_scope(ctx, "web_search.anthropic"),
                timeout=_web_search_transport_timeout(ctx),
                processing_preference=processing_preference,
            )
        blocks = _obj_to_plain(getattr(response, "content", []) or [])
        text_parts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and str(block.get("type") or "") == "text":
                    text_parts.append(str(block.get("text") or ""))
        usage = _obj_to_plain(getattr(response, "usage", None))
        if not isinstance(usage, dict):
            usage = {}
        from ouroboros.llm_attempt import attach_processing_receipt

        attach_processing_receipt({"provider": "anthropic", "usage_model": active_model}, usage)
        usage["ledger_attempt_ids"] = list(attempt_ids)
        _emit_simple_usage(
            ctx,
            provider="anthropic",
            model=active_model,
            prompt_tokens=int((usage or {}).get("input_tokens") or 0) if isinstance(usage, dict) else 0,
            completion_tokens=int((usage or {}).get("output_tokens") or 0) if isinstance(usage, dict) else 0,
            usage=usage if isinstance(usage, dict) else {},
        )
        return json.dumps({
            "answer": "".join(text_parts).strip() or "(no answer)",
            "answer_type": "summary",
            "sources": _extract_sources_from_response(response),
            "backend": "anthropic_server_tool",
            "model": active_model, "reported_model": getattr(response, "model", None), "ledger_attempt_ids": list(attempt_ids),
        }, ensure_ascii=False, indent=2)
    except UsageAccountingError:
        raise
    except Exception as exc:
        raise _wrapped_provider_error("Anthropic", exc) from exc


def _web_search_ddgs(query: str, *, _max_attempts: int = 3) -> str:
    # ddgs is an unofficial scraper with no SLA: it raises a RatelimitException
    # under sustained load. Retry a few times with backoff on transient rate-limit/
    # timeout errors so a full benchmark run (many sequential searches) survives.
    last_exc: Exception | None = None
    for attempt in range(max(1, _max_attempts)):
        try:
            from ddgs import DDGS

            with DDGS() as ddgs_client:
                results = list(ddgs_client.text(query, max_results=10))
            sources = [{
                "url": sanitize_tool_result_for_log(str(item.get("href") or item.get("url") or "")),
                "title": sanitize_tool_result_for_log(str(item.get("title") or "")),
                "snippet": sanitize_tool_result_for_log(str(item.get("body") or item.get("snippet") or "")),
            } for item in results]
            answer = "\n".join(
                f"- {item['title']}: {item['snippet']} ({item['url']})"
                for item in sources
                if item["url"] or item["snippet"]
            )
            return json.dumps({
                "answer": answer or "(no answer)",
                "answer_type": "summary",
                "sources": sources,
                "backend": "ddgs",
            }, ensure_ascii=False, indent=2)
        except Exception as exc:
            last_exc = exc
            name = type(exc).__name__.casefold()
            msg = str(exc).casefold()
            transient = ("ratelimit" in name or "ratelimit" in msg or "429" in msg
                         or "202" in msg or "timeout" in name)
            if transient and attempt + 1 < _max_attempts:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    detail = sanitize_tool_result_for_log(str(last_exc))[:500]
    raise RuntimeError(f"DDGS web search failed ({type(last_exc).__name__}): {detail}") from last_exc


def _is_timeout_error(exc: Exception) -> bool:
    """Typed timeout classifier across the installed HTTP/SDK transports."""
    if isinstance(exc, TimeoutError):
        return True
    # Keep the OpenAI timeout contract available when the optional SDK class
    # cannot be imported in the caller's environment (and for compatible SDK
    # wrappers that preserve the public exception type in their MRO).
    if any(cls.__name__ == "APITimeoutError" for cls in type(exc).__mro__):
        return True
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return True
    except Exception:
        pass
    try:
        from openai import APITimeoutError

        return isinstance(exc, APITimeoutError)
    except Exception:
        return False


def _provider_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if not isinstance(value, int):
        value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def _stream_failure_error(event_type: str, error: Any, response: Any) -> RuntimeError:
    """Preserve a terminal Responses-event status for the safe retry policy."""
    detail = sanitize_tool_result_for_log(
        str(getattr(error, "message", None) or error or "no detail")
    )[:300]
    exc = RuntimeError(f"OpenAI web search {event_type}: {detail}")
    for obj in (error, getattr(response, "error", None), response):
        for field in ("status_code", "http_status", "status", "code"):
            value = obj.get(field) if isinstance(obj, dict) else getattr(obj, field, None)
            try:
                status = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= status <= 599:
                setattr(exc, "status_code", status)
                return exc
    return exc


def _responses_search_candidate(ctx, query: str, model: str, options: dict):
    """One complete Responses body supplies capture, reservation and actual send."""
    from ouroboros.llm_attempt import (
        _attempt_request, _candidate_before_dispatch, _finalized_physical_candidate,
        apply_processing_preference,
    )
    target = {"provider": "openai", "resolved_model": model, "base_url": options["base_url"],
              "usage_model": model if "/" in model else f"openai/{model}",
              "processing_preference": options["preference"], "requested_reasoning_effort": options["effort"]}
    payload = {"model": model,
               "tools": [{"type": "web_search", "search_context_size": options["search_context_size"]}],
               "reasoning": {"effort": options["effort"]}, "tool_choice": "auto", "input": query,
               "stream": True}
    apply_processing_preference(target, payload)
    if options.get("submission") == "standard":
        payload["service_tier"] = "default"
    candidate = _finalized_physical_candidate(target, payload, "responses")
    scope = _accounting_scope(ctx, "web_search.openai_responses")
    request = replace(_attempt_request(target, candidate),
        max_completion_tokens=8192,
        drive_root=scope.drive_root,
        task_id=scope.task_id,
        root_task_id=scope.root_task_id,
        parent_task_id=scope.parent_task_id,
        category=scope.category,
        source=scope.source,
    )
    return target, candidate, request, _candidate_before_dispatch(candidate, request)


def _web_search_openai(
    ctx: ToolContext,
    query: str,
    model: str = "",
    search_context_size: str = "",
    reasoning_effort: str = "",
    _attempt: int = 0,
    _processing_preference: Optional[str] = None,
    _processing_submission: Optional[str] = None,
) -> str:
    from ouroboros.model_slots import resolve_processing_preference

    preference = resolve_processing_preference("websearch", override=_processing_preference)
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    def _fallbacks(previous_errors=None):
        return json.dumps({"error": "; ".join(previous_errors or ["Official OpenAI search is unavailable"]),
                           "backend": "openai_responses", "ledger_attempt_ids": attempt_ids})
    attempt_ids = []
    api_key, base_url, provider, api_key_type = _resolve_openai_client_settings()
    if not api_key:
        return _fallbacks()
    active_model = model or runtime_setting("OUROBOROS_WEBSEARCH_MODEL", DEFAULT_SEARCH_MODEL)
    active_context = search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE
    active_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    reservation = None
    dispatched = False
    send_started = False
    explicit_provider_failure = False
    manifest = None
    try:
        from ouroboros.config import get_finalization_grace_sec
        from ouroboros.net_transport import web_search_openai_client
        transport_timeout = _web_search_transport_timeout(ctx)
        # Explicit per-attempt transport bound; the outer ToolEntry covers the
        # configured paid cascade rather than acting as the socket timeout.
        client = web_search_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout=transport_timeout,
        )
        target, candidate, request, before_dispatch = _responses_search_candidate(ctx, query, active_model, {
            "base_url": base_url, "preference": preference, "submission": _processing_submission,
            "search_context_size": active_context, "effort": active_effort})
        reservation = reserve_attempt(request)
        attempt_ids.append(reservation.attempt_id)
        if _web_search_deadline_exhausted(ctx):
            release_attempt(reservation, "deadline_exhausted_before_dispatch")
            reservation = None
            return _web_search_deadline_result()
        manifest = before_dispatch(reservation)
        mark_dispatched(reservation, candidate_manifest_ref=manifest)
        dispatched = True
        try:
            send_started = True
            stream = client.responses.create(**candidate)
        except Exception as error:
            from ouroboros.llm_attempt import processing_refusal

            refusal = processing_refusal(target, candidate, error)
            if refusal is error:
                raise
            raise refusal from error
        text_parts: list[str] = []
        usage: dict = {}
        sources: List[Dict[str, str]] = []
        progress_sent = False
        response_completed = False
        reported_cost, reported_cost_final = None, False

        for event in stream:
            etype = getattr(event, "type", "")

            # Provider-side failure events must NOT be swallowed: an errored or
            # incomplete response would otherwise fall through to the "(no
            # answer)" success return below, so the OpenAI leg "succeeds" empty
            # and the OpenRouter/Anthropic/ddgs cascade never engages.
            if etype in ("response.failed", "error", "response.incomplete"):
                explicit_provider_failure = True
                resp_obj = getattr(event, "response", None)
                err = getattr(event, "error", None) or (getattr(resp_obj, "error", None) if resp_obj else None)
                raise _stream_failure_error(etype, err, resp_obj)

            # Web search lifecycle — emit progress so the user sees activity
            if etype in (
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
            ) and not progress_sent:
                if hasattr(ctx, "emit_progress_fn") and ctx.emit_progress_fn:
                    try:
                        safe_query = sanitize_tool_result_for_log(str(query or ""))[:100]
                        ctx.emit_progress_fn(f"🔍 Searching: {safe_query}")
                    except Exception:
                        pass
                progress_sent = True

            # Accumulate text deltas
            elif etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    text_parts.append(delta)

            # Final event — extract usage for cost tracking
            elif etype == "response.completed":
                response_completed = True
                resp_obj = getattr(event, "response", None)
                if resp_obj:
                    from ouroboros._usage_response import processing_receipt, usage_from_response

                    usage, reported_cost, reported_cost_final = usage_from_response(resp_obj)
                    receipt = processing_receipt("openai", usage,
                        requested=request.processing_preference, submitted_native=request.submitted_processing_mode)
                    if receipt is not None:
                        usage["processing"] = receipt
                    sources = _extract_sources_from_response(resp_obj)

        text = "".join(text_parts)
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        usage["ledger_attempt_ids"] = [reservation.attempt_id]
        try:
            if response_completed:
                settle_attempt(reservation, usage, cost_usd=reported_cost,
                               cost_final=reported_cost_final)
            else:
                mark_unresolved(reservation, "Responses stream ended without response.completed")
            dispatched = False
        except Exception as exc:
            # Preserve an already-paid result; unresolved retains its upper bound.
            log.exception("Failed to settle OpenAI Responses search attempt")
            try:
                mark_unresolved(reservation, f"settlement_write_failed:{type(exc).__name__}")
                dispatched = False
            except Exception:
                log.exception("Failed to mark Responses settlement unresolved")

        if not response_completed:
            return json.dumps({
                "error": (
                    "OpenAI web search ended without a terminal provider outcome; "
                    "no retry or paid fallback was sent."
                ),
                "backend": "openai_responses",
                "reason_code": "provider_outcome_unknown",
                "ledger_attempt_ids": attempt_ids,
            }, ensure_ascii=False, indent=2)

        # Track web search cost (estimate from tokens — OpenAI usage has no total_cost)
        if usage and hasattr(ctx, "pending_events"):
            cost = _estimate_openai_cost(active_model, input_tokens, output_tokens)
            metadata = getattr(ctx, "task_metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            try:
                ctx.pending_events.append({
                    "type": "llm_usage",
                    "task_id": str(getattr(ctx, "task_id", "") or ""),
                    "root_task_id": str(metadata.get("root_task_id") or ""),
                    "parent_task_id": str(metadata.get("parent_task_id") or ""),
                    "delegation_role": str(metadata.get("delegation_role") or ""),
                    "provider": provider,
                    "model": active_model,
                    "api_key_type": api_key_type,
                    "model_category": "websearch",
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "usage": usage,
                    "cost": cost,
                    "source": "web_search",
                    "ts": utc_now_iso(),
                    "category": "task",
                    "accounting_authority": "physical_attempt_ledger",
                    "ledger_attempt_ids": [reservation.attempt_id],
                })
            except Exception:
                log.debug("Failed to emit web_search cost event", exc_info=True)

        if text.strip() or sources:
            return json.dumps({"answer": text or "(no answer)", "answer_type": "summary", "sources": sources, "backend": "openai_responses", "model": active_model, "reported_model": getattr(resp_obj, "model", None), "ledger_attempt_ids": attempt_ids}, ensure_ascii=False, indent=2)
    except Exception as e:
        from ouroboros.llm_attempt import ProcessingNotStarted

        processing_refused = isinstance(e, ProcessingNotStarted)
        proven_not_started = False
        if dispatched:
            from ouroboros.transport_custody import release_pre_dispatch_attempt

            proven_not_started = release_pre_dispatch_attempt(reservation, e)
            dispatched = not proven_not_started
        was_dispatched = send_started and not proven_not_started
        if reservation is not None and dispatched:
            try:
                mark_unresolved(reservation, f"{type(e).__name__}: {e}")
            except Exception:
                log.exception("Failed to mark OpenAI Responses search unresolved")
        if reservation is not None and (not send_started or proven_not_started):
            from ouroboros.usage_accounting import _pre_dispatch_failure

            failure = _pre_dispatch_failure(reservation, request, e, candidate_manifest_ref=manifest)
            if isinstance(e, UsageAccountingError) or failure.physical_attempt_capture.state != "released":
                raise failure from e
        if isinstance(e, UsageAccountingError):
            raise
        detail = sanitize_tool_result_for_log(str(e))[:500]
        status = _provider_status_code(e)
        terminal_provider_failure = explicit_provider_failure or status is not None
        if was_dispatched and not terminal_provider_failure:
            return json.dumps({
                "error": (
                    "OpenAI web search was dispatched but its provider outcome is "
                    "unknown; no retry or paid fallback was sent."
                ),
                "backend": "openai_responses",
                "reason_code": "provider_outcome_unknown",
                "ledger_attempt_ids": attempt_ids,
            }, ensure_ascii=False, indent=2)
        retryable_terminal = status == 408 or status == 429 or (
            status is not None and 500 <= status <= 599
        )
        # One retry is safe only before dispatch or after an explicit terminal
        # provider response. An ambiguous dispatched outcome stops the cascade.
        if _attempt == 0 and (
            (_is_timeout_error(e) and not was_dispatched) or retryable_terminal or processing_refused
        ):
            from ouroboros.deadline_utils import deadline_remaining_sec, has_deadline

            if has_deadline(ctx) and deadline_remaining_sec(ctx) <= max(
                1.0, float(get_finalization_grace_sec())
            ):
                return _fallbacks([f"OpenAI web search timed out before a safe retry: {detail}"])
            log.debug("web_search OpenAI safe terminal/pre-dispatch retry")
            return _web_search_openai(
                ctx, query, model=model, search_context_size=search_context_size,
                reasoning_effort=reasoning_effort, _attempt=1,
                _processing_preference=preference,
                _processing_submission="standard" if processing_refused else _processing_submission,
            )
        return _fallbacks([f"OpenAI web search failed ({type(e).__name__}): {detail}"])
    # An empty completed result is a soft failure. Leave this backend's
    # cleanup/capture scope before the cascade so another backend's exception
    # cannot release this paid attempt or return its physical claim.
    return _fallbacks(["OpenAI web search returned no answer and no sources"])


def _web_search(ctx: ToolContext, query: str, model: str = "", search_context_size: str = "",
                reasoning_effort: str = "") -> str:
    from ouroboros.model_slots import resolve_processing_preference

    route = resolve_web_search_route(model=model or None)
    tried = []
    preference = resolve_processing_preference("websearch")
    result = {"error": route.get("error") or f"web_search unavailable: source {route['source']} has no configured search backend that succeeded. Configure a search API key or install optional ddgs.",
              "backend": route["source"]}
    # Auto tries eligible routes; a pinned source always reaches its own backend,
    # which reports its own missing credential or accounting refusal.
    for leg in route["legs"] + ([] if route["source"] == "auto" else route["unavailable"]):
        if _web_search_deadline_exhausted(ctx):
            result = json.loads(_web_search_deadline_result())
            break
        try:
            with capture_attempt_ids() as ids:
                if leg["source"] == "openai":
                    raw = _web_search_openai(ctx, query, model=leg["model"], search_context_size=search_context_size,
                                             reasoning_effort=reasoning_effort, _processing_preference=preference)
                elif leg["source"] == "openrouter":
                    raw = _web_search_openrouter(ctx, query, model=leg["model"], search_context_size=search_context_size,
                                                 processing_preference=preference)
                elif leg["source"] == "anthropic":
                    raw = _web_search_anthropic(ctx, query, model="anthropic::" + leg["model"], processing_preference=preference)
                else:
                    raw = _web_search_ddgs(query)
            result = json.loads(raw)
        except UsageAccountingError:
            raise
        except Exception as exc:
            result = (json.loads(_unknown_provider_outcome(leg["backend"])) if _provider_outcome_is_unknown(exc)
                      else {"error": sanitize_tool_result_for_log(str(exc))[:500]})
            capture = getattr(exc, "physical_attempt_capture", None)
            if capture and getattr(capture, "attempt_id", None):
                ids.append(capture.attempt_id)
            ids.extend(getattr(exc, "ledger_attempt_ids", []) or [])
        # A pinned source's failure is reported as that source; legs_tried names the backend.
        pinned_failure = route["source"] != "auto" and "error" in result and result.get("reason_code") is None
        result.update(backend=route["source"] if pinned_failure else leg["backend"], model=leg["model"])
        tried.append({**leg, "outcome": result.get("reason_code") or ("failed" if "error" in result else "succeeded"),
                      "ledger_attempt_ids": list(dict.fromkeys([*ids, *result.get("ledger_attempt_ids", [])])),
                      **({"error": result["error"]} if "error" in result else {})})
        if "error" not in result or result.get("reason_code") in {"provider_outcome_unknown", "deadline_exhausted"}:
            break
    return json.dumps({**result, "route": route, "legs_tried": tried,
                       "backend_errors": [leg["error"] for leg in tried if "error" in leg]}, ensure_ascii=False, indent=2)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                "Search the web with the configured Source and Model. Auto uses the eligible built-in "
                "routes listed in this task's runtime facts; a specific source is strict. "
                "Skills, MCP and browser tools are independent choices. A per-call model override "
                "follows the same source selection. Empty model uses the saved choice or provider defaults. "
                "For a COMPOUND question (several distinct facts/entities/time ranges in one ask), "
                "issue one focused web_search per sub-question instead of one broad query — "
                "narrow queries return sharper sources. These read-only searches run in parallel. "
                "The returned `answer` is a SUMMARY/lead (answer_type=summary), not a primary source: "
                "confirm load-bearing facts by opening the returned sources (browse_page) before "
                "relying on them."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "A single focused search query (split compound asks into separate calls)."},
                "model": {"type": "string", "description": "Search model override; native id or routed source::model. Must agree with a pinned Source."},
                "search_context_size": {"type": "string", "enum": ["low", "medium", "high"],
                                        "description": f"How much context to fetch (default: {DEFAULT_SEARCH_CONTEXT_SIZE})"},
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"],
                                     "description": f"Reasoning effort (default: {DEFAULT_REASONING_EFFORT})"},
            }, "required": ["query"]},
        }, _web_search, timeout_sec=_web_search_outer_timeout_sec()),
    ]
