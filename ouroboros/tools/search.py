"""Web search tool — OpenAI Responses API with LLM-first overridable defaults."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List

from ouroboros.pricing import estimate_cost
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso

log = logging.getLogger(__name__)

DEFAULT_SEARCH_MODEL = "gpt-5.2"
DEFAULT_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_REASONING_EFFORT = "high"

def _estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost through the shared pricing table."""
    pricing_model = model if "/" in str(model or "") else f"openai/{model}"
    cost = estimate_cost(pricing_model, input_tokens, output_tokens)
    if cost:
        return cost
    return round(input_tokens * 2.0 / 1_000_000 + output_tokens * 10.0 / 1_000_000, 6)


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
    official_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    legacy_base_url = (os.environ.get("OPENAI_BASE_URL", "") or "").strip()

    if official_key and not legacy_base_url:
        return official_key, None, "openai", "openai"
    return "", None, "openai", "openai"


def _openrouter_model(model: str) -> str:
    active = str(model or os.environ.get("OUROBOROS_WEBSEARCH_MODEL") or DEFAULT_SEARCH_MODEL).strip()
    if not active:
        active = DEFAULT_SEARCH_MODEL
    return active if "/" in active else f"openai/{active}"


def _anthropic_model(model: str) -> str:
    active = str(model or os.environ.get("OUROBOROS_WEBSEARCH_MODEL") or "").strip()
    if active.startswith("anthropic::"):
        return active[len("anthropic::"):]
    if active.startswith("anthropic/"):
        return active[len("anthropic/"):]
    return "claude-sonnet-4-6"


def _available_web_search_backends() -> list[str]:
    backends: list[str] = []
    openai_key, _base_url, _provider, _api_key_type = _resolve_openai_client_settings()
    if openai_key:
        backends.append("openai_responses")
    if str(os.environ.get("OPENROUTER_API_KEY") or "").strip():
        backends.append("openrouter_server_tool")
    if str(os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        backends.append("anthropic_server_tool")
    try:
        import ddgs  # noqa: F401

        backends.append("ddgs")
    except Exception:
        pass
    return backends


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
        cost = estimate_cost(model if "/" in str(model) else f"{provider}/{model}", prompt_tokens, completion_tokens)
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
        })
    except Exception:
        log.debug("Failed to emit web_search fallback cost event", exc_info=True)


def _web_search_openrouter(ctx: ToolContext, query: str, model: str = "", search_context_size: str = "") -> str:
    api_key = str(os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    try:
        from ouroboros.llm import openrouter_web_search_server_tool

        active_model = _openrouter_model(model)
        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Searching via OpenRouter: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        response = openrouter_web_search_server_tool(
            api_key=api_key,
            model=active_model,
            query=query,
            search_context_size=search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE,
        )
        message = response.choices[0].message if getattr(response, "choices", None) else None
        text = str(getattr(message, "content", "") or "").strip()
        usage_obj = getattr(response, "usage", None)
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else (_obj_to_plain(usage_obj) if usage_obj else {})
        # Guard: an exotic (non-dict) usage object must not crash the leg with a
        # successful paid answer in hand — mirror the Anthropic leg's isinstance
        # check (a `.get` on a non-dict would raise and discard the result).
        if not isinstance(usage, dict):
            usage = {}
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
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        detail = sanitize_tool_result_for_log(str(exc))[:500]
        raise RuntimeError(f"OpenRouter web search failed ({type(exc).__name__}): {detail}") from exc


def _web_search_anthropic(ctx: ToolContext, query: str, model: str = "") -> str:
    api_key = str(os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    try:
        from ouroboros.llm import anthropic_web_search_server_tool

        active_model = _anthropic_model(model)
        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Searching via Anthropic: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        response = anthropic_web_search_server_tool(
            api_key=api_key,
            model=active_model,
            query=query,
        )
        blocks = _obj_to_plain(getattr(response, "content", []) or [])
        text_parts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and str(block.get("type") or "") == "text":
                    text_parts.append(str(block.get("text") or ""))
        usage = _obj_to_plain(getattr(response, "usage", None))
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
        }, ensure_ascii=False, indent=2)
    except Exception as exc:
        detail = sanitize_tool_result_for_log(str(exc))[:500]
        raise RuntimeError(f"Anthropic web search failed ({type(exc).__name__}): {detail}") from exc


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
    """Heuristic-free timeout classifier: real timeout exception types only."""
    if isinstance(exc, TimeoutError):
        return True
    return "timeout" in type(exc).__name__.casefold()


def _web_search(
    ctx: ToolContext,
    query: str,
    model: str = "",
    search_context_size: str = "",
    reasoning_effort: str = "",
    _attempt: int = 0,
) -> str:
    # Backend pin: force ONE backend regardless of which provider keys are present.
    # A fixed-model run pins 'ddgs' so web_search is pure-retrieval (no second LLM
    # contaminating the "single fixed model" claim). 'openai' pins the OpenAI leg only
    # (no cascade — see _fallbacks). 'auto'/'' keep the default OpenAI-first cascade
    # below. This is a config gate on TRANSPORT, not on agent behaviour (P5-safe).
    pinned = (os.environ.get("OUROBOROS_WEBSEARCH_BACKEND") or "").strip().lower()
    if pinned in ("ddgs", "openrouter", "anthropic"):
        try:
            if pinned == "ddgs":
                return _web_search_ddgs(query)
            if pinned == "openrouter":
                return _web_search_openrouter(ctx, query, model=model, search_context_size=search_context_size)
            return _web_search_anthropic(ctx, query, model=model)
        except Exception as exc:
            detail = sanitize_tool_result_for_log(str(exc))[:500]
            return json.dumps(
                {"error": f"pinned web_search backend '{pinned}' failed: {detail}", "backend": pinned},
                ensure_ascii=False, indent=2,
            )

    def _fallbacks(previous_errors: list[str] | None = None) -> str:
        errors = list(previous_errors or [])
        if pinned == "openai":
            # 'openai' is a TRUE pin: hard-fail rather than cascading to other backends,
            # so a fixed/repro run cannot silently fall back to a different transport.
            detail = "; ".join(errors) if errors else (
                "no official OPENAI_API_KEY (without OPENAI_BASE_URL) configured"
            )
            return json.dumps(
                {"error": f"pinned web_search backend 'openai' unavailable: {detail}", "backend": "openai"},
                ensure_ascii=False, indent=2,
            )
        for backend in (
            lambda: _web_search_openrouter(ctx, query, model=model, search_context_size=search_context_size),
            lambda: _web_search_anthropic(ctx, query, model=model),
            lambda: _web_search_ddgs(query),
        ):
            try:
                return backend()
            except Exception as exc:
                errors.append(sanitize_tool_result_for_log(str(exc))[:500])
        return json.dumps({
            "error": (
                "web_search unavailable: no configured search backend succeeded. "
                "Configure official OPENAI_API_KEY (without OPENAI_BASE_URL), OPENROUTER_API_KEY, "
                "ANTHROPIC_API_KEY, or install optional ddgs."
            ),
            "backend_errors": errors,
        }, ensure_ascii=False, indent=2)

    api_key, base_url, provider, api_key_type = _resolve_openai_client_settings()
    if not api_key:
        return _fallbacks()

    active_model = model or os.environ.get("OUROBOROS_WEBSEARCH_MODEL", DEFAULT_SEARCH_MODEL)
    active_context = search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE
    active_effort = reasoning_effort or DEFAULT_REASONING_EFFORT

    try:
        from openai import OpenAI
        from ouroboros.config import get_websearch_timeout_sec
        # Explicit transport timeout (v6.54.3, D): without it the streaming SDK
        # call had NO client bound, so the ToolEntry 540s outer thread-kill was
        # the only stop for a wedged stream.
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=get_websearch_timeout_sec())

        # --- Streaming path: emit progress while the search runs ---
        stream = client.responses.create(
            model=active_model,
            tools=[{
                "type": "web_search",
                "search_context_size": active_context,
            }],
            reasoning={"effort": active_effort},
            tool_choice="auto",
            input=query,
            stream=True,
        )

        text_parts: list[str] = []
        usage: dict = {}
        sources: List[Dict[str, str]] = []
        progress_sent = False

        for event in stream:
            etype = getattr(event, "type", "")

            # Provider-side failure events must NOT be swallowed: an errored or
            # incomplete response would otherwise fall through to the "(no
            # answer)" success return below, so the OpenAI leg "succeeds" empty
            # and the OpenRouter/Anthropic/ddgs cascade never engages.
            if etype in ("response.failed", "error", "response.incomplete"):
                resp_obj = getattr(event, "response", None)
                detail = ""
                err = getattr(event, "error", None) or (getattr(resp_obj, "error", None) if resp_obj else None)
                if err is not None:
                    detail = sanitize_tool_result_for_log(str(getattr(err, "message", None) or err))[:300]
                raise RuntimeError(f"OpenAI web search {etype}: {detail or 'no detail'}")

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
                resp_obj = getattr(event, "response", None)
                if resp_obj:
                    u = getattr(resp_obj, "usage", None)
                    if u:
                        usage = u.model_dump() if hasattr(u, "model_dump") else {}
                    sources = _extract_sources_from_response(resp_obj)

        text = "".join(text_parts)

        # An empty result (no answer text AND no sources) is a soft failure, not
        # a successful "(no answer)": fall through to the provider cascade so a
        # degenerate OpenAI response does not shadow a working OpenRouter/
        # Anthropic/ddgs backend.
        if not text.strip() and not sources:
            return _fallbacks(["OpenAI web search returned no answer and no sources"])

        # Track web search cost (estimate from tokens — OpenAI usage has no total_cost)
        if usage and hasattr(ctx, "pending_events"):
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
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
                })
            except Exception:
                log.debug("Failed to emit web_search cost event", exc_info=True)

        return json.dumps({"answer": text or "(no answer)", "answer_type": "summary", "sources": sources, "backend": "openai_responses"}, ensure_ascii=False, indent=2)
    except Exception as e:
        detail = sanitize_tool_result_for_log(str(e))[:500]
        # One retry on a genuine timeout before cascading: web search timeouts are
        # frequently transient, and the provider cascade is slower/less precise.
        if _attempt == 0 and _is_timeout_error(e):
            log.debug("web_search OpenAI timeout; retrying once")
            return _web_search(
                ctx, query, model=model, search_context_size=search_context_size,
                reasoning_effort=reasoning_effort, _attempt=1,
            )
        return _fallbacks([f"OpenAI web search failed ({type(e).__name__}): {detail}"])


def get_tools() -> List[ToolEntry]:
    backends = _available_web_search_backends()
    backend_note = ", ".join(backends) if backends else "unavailable (no key/backend configured)"
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                "Search the web using the best available backend "
                f"({backend_note}). Preferred order: OpenAI Responses, OpenRouter server tool, "
                "Anthropic server tool, optional ddgs. "
                f"Defaults: model={DEFAULT_SEARCH_MODEL}, search_context_size={DEFAULT_SEARCH_CONTEXT_SIZE}, "
                f"reasoning_effort={DEFAULT_REASONING_EFFORT}. "
                "Override any parameter per-call if needed (LLM-first: you decide). "
                "For a COMPOUND question (several distinct facts/entities/time ranges in one ask), "
                "issue one focused web_search per sub-question instead of one broad query — "
                "narrow queries return sharper sources. These read-only searches run in parallel. "
                "The returned `answer` is a SUMMARY/lead (answer_type=summary), not a primary source: "
                "confirm load-bearing facts by opening the returned sources (browse_page) before "
                "relying on them."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "A single focused search query (split compound asks into separate calls)."},
                "model": {"type": "string", "description": f"OpenAI model (default: {DEFAULT_SEARCH_MODEL})"},
                "search_context_size": {"type": "string", "enum": ["low", "medium", "high"],
                                        "description": f"How much context to fetch (default: {DEFAULT_SEARCH_CONTEXT_SIZE})"},
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"],
                                     "description": f"Reasoning effort (default: {DEFAULT_REASONING_EFFORT})"},
            }, "required": ["query"]},
        }, _web_search, timeout_sec=540),
    ]
