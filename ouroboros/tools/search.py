"""Web search accounts each physical provider attempt. Failed Responses, OpenRouter and Anthropic attempts retry or
cascade only after a proven no-send or a typed no-start refusal without contrary evidence. A router's status and words
(OpenRouter's free 404 "No endpoints found") make another source possible, not supported. A confirmed empty Responses
completion may advance Auto as a separate leg. DeepSeek follows its Search API contract. DDGS is an unofficial scraper.
A leg's ``llm_usage`` event states the amount its attempt row settled, final or
the ledger's own estimate, that row's ``cost_invalid`` and its token counts; the tool prices and counts nothing itself."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from ouroboros._usage_response import refusal_conflicts, refusal_failure_payload, typed_search_refusal
from ouroboros.gateways import search_api
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.transport_custody import _capture_on_chain, is_processing_refusal, release_pre_dispatch_attempt, stream_incomplete_on_chain
from ouroboros.usage_accounting import (
    AttemptRequest,
    PhysicalAttemptCapture,
    UsageAccountingError,
    UsageScope,
    capture_attempt_ids,
    current_usage_scope,
    execute_physical_attempt,
    mark_dispatched,
    mark_unresolved,
    release_attempt,
    reserve_attempt,
    settle_attempt,
)
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso
from ouroboros.config import runtime_setting

log = logging.getLogger(__name__)

from ouroboros.search_routes import MODELLESS_SOURCES, SEARCH_MODELS, named_connection_routes, resolve_web_search_route

DEFAULT_SEARCH_MODEL = SEARCH_MODELS["openai"]
DEFAULT_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_REASONING_EFFORT = "high"


def _wrapped_provider_error(provider: str, exc: Exception) -> RuntimeError:
    """Sanitize a provider error without dropping its physical-attempt fact."""
    detail = sanitize_tool_result_for_log(str(exc))[:500]
    wrapped = RuntimeError(f"{provider} web search failed ({type(exc).__name__}): {detail}")
    wrapped.ledger_attempt_ids = list(getattr(exc, "ledger_attempt_ids", []) or [])
    capture = _capture_on_chain(exc)
    if isinstance(capture, PhysicalAttemptCapture):
        setattr(wrapped, "physical_attempt_capture", capture)
    return wrapped


def _provider_outcome_is_unknown(exc: Exception) -> bool:
    """Whether another leg might repeat a search. Settlement alone says nothing about that."""
    capture = _capture_on_chain(exc)
    if getattr(capture, "provider", None) == "deepseek":
        return stream_incomplete_on_chain(exc, exclude_rejected=True) or (
            capture.state in {"dispatched", "unresolved"} and capture.provider_status_code is None)
    return refusal_conflicts(exc)


def _unknown_provider_outcome(backend: str, attempt_ids=()) -> str:
    return json.dumps({
        "error": (
            f"{backend} web search was dispatched but its provider outcome is unknown; "
            "no retry or paid fallback was sent."
        ),
        "backend": backend,
        "reason_code": "provider_outcome_unknown",
        "ledger_attempt_ids": list(attempt_ids),
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


def _web_search_caller_deadline(ctx: Any) -> dict:
    """The same deadline less reserve as the recovery driver's absolute bound: its preparation,
    slot wait and every physical send re-check it, so recording time cannot ride into the reserve."""
    from ouroboros.deadline_utils import caller_deadline_arguments
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    metadata = getattr(ctx, "task_metadata", {})
    deadline_at = metadata.get("deadline_at") if isinstance(metadata, dict) else None
    return caller_deadline_arguments(deadline_at, None, reserve_sec=effective_finalization_reserve_sec(ctx),
                                     dispatch_only=True)


def _web_search_deadline_exhausted(ctx: Any) -> bool:
    from ouroboros.deadline_utils import owner_deadline_exhausted_for_context
    from ouroboros.task_pacing import effective_finalization_reserve_sec
    return owner_deadline_exhausted_for_context(ctx, reserve_sec=effective_finalization_reserve_sec(ctx))


def _own_window_refused(ctx: Any, error: BaseException) -> bool:
    """An unsent refusal by this leg's own deadline-less-reserve window, which its caller loop then finalizes after
    as after any exhausted leg; the task's own controls and every possibly sent attempt keep their rails (#572)."""
    from ouroboros.llm_attempt import refused_by_own_window

    return refused_by_own_window(error, own_window_spent=_web_search_deadline_exhausted(ctx))


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


def _search_target(kind: str, connection: str, model: str) -> dict:
    """The resolved target the search leg sends ``model`` on, read once through the one resolver: its connection's
    endpoint, credential, headers, billing identity (``usage_model``) and provider routing; {} when it cannot serve."""
    from ouroboros.llm_routing import resolve_target
    from ouroboros.provider_models import connection_id, legacy_openai_base

    if kind == "openai" and not connection and legacy_openai_base():
        return {}  # The legacy key belongs to that compatible endpoint, never to official Responses.

    try:
        target = resolve_target(f"{connection_id(kind, connection)}::{model}")
    except Exception:
        return {}
    if kind == "anthropic" and not connection and (legacy_base := os.environ.get("ANTHROPIC_BASE_URL")):
        # The built-in SDK search historically uses this prefix. Named keys belong to their resolved endpoint.
        base = legacy_base.rstrip("/") + "/v1"
        target = {**target, "base_url": base, "capability_route": {**target["capability_route"], "dispatch_base_url": base}}
    key = str(target.get("api_key") or "").strip()
    return {**target, "api_key": key} if key else {}


def _search_key(kind: str, connection: str = "") -> str:
    """The key of the search leg's connection ('' when it cannot serve)."""
    return str(_search_target(kind, connection, "search").get("api_key") or "")


def _leg_model(source: str, model: str) -> str:
    route = resolve_web_search_route(backend=source, model=model or None)
    legs = route["legs"] + route["unavailable"]
    if not legs:
        raise ValueError(route.get("error", "Search model unavailable"))
    return legs[0]["model"]


def _available_web_search_backends() -> list[str]:
    """Backends a call could reach: the configured selection's eligible legs, else the Source's recommended
    ones (a call naming its own model meets them, so not under a Source that takes none), else a named
    connection's, which only a call naming it uses."""
    route = resolve_web_search_route()
    legs = (route["legs"] or ([] if route["source"] in MODELLESS_SOURCES else resolve_web_search_route(model="")["legs"])
            or [leg for named in named_connection_routes() for leg in named["legs"]])
    return [leg["backend"] for leg in legs]


def _web_search_unavailability() -> tuple[str, str]:
    """``(reason, detail)`` when no call can run, from the routes' own causes: the configured one's,
    then each configured named connection's (``kind@name``), whose key may be there without its SDK.

    Only a route whose every unavailable leg lacks a key is ``missing_credential``;
    a key whose SDK is not installed keeps ``package_missing``. An unsupported saved
    selection is named beside what also stops the Source's recommended legs, or alone
    under a Source that takes no model, where no call reaches those legs."""
    from ouroboros.provider_models import connection_id

    route = resolve_web_search_route()
    selection_error = route.get("error", "")
    if selection_error:
        route = resolve_web_search_route(model="")
        if route.get("error") or route["source"] in MODELLESS_SOURCES:
            return "unsupported_selection", selection_error
    unavailable = route["unavailable"] + [leg for named in named_connection_routes() for leg in named["unavailable"]]
    reasons = list(dict.fromkeys("missing_credential" if leg["unavailable_reason"] == "no_credentials"
                                 else leg["unavailable_reason"] for leg in unavailable)) or ["missing_credential"]
    causes = ", ".join(f"{connection_id(leg['source'], leg.get('connection', ''))} {leg['unavailable_reason']}"
                       for leg in unavailable)
    detail = f"web_search_backend: {causes}" if causes else "web_search_backend"
    if selection_error:
        return "+".join(["unsupported_selection", *reasons]), f"{selection_error}; {detail}"
    return "+".join(reasons), detail


def _web_search_backend_pin() -> str:
    return (runtime_setting("OUROBOROS_WEBSEARCH_BACKEND") or "").strip().lower()


def _web_search_outer_timeout_sec() -> int:
    """Cover every configured paid leg absent a tighter owner deadline."""
    from ouroboros.config import get_finalization_grace_sec, get_websearch_timeout_sec

    pinned = _web_search_backend_pin()
    if pinned == "openai":
        attempts = 2
    elif pinned in {"openrouter", "anthropic", "deepseek"}:
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


def _settled_fact(attempt_ids: List[str], drive_root: Any = None) -> Dict[str, Any]:
    """The ledger's money and counts of the attempt that carried a leg's reply (its last id). ``drive_root`` is
    that reservation's own; a leg sent through the shared driver reads it from this context's last capture, which
    must be that very attempt's. Empty when either is missing."""
    from ouroboros._usage_settlement import attempt_money_fact
    from ouroboros.usage_accounting import current_physical_attempt_drive_root, last_physical_attempt_capture

    if not attempt_ids:
        return {}
    if drive_root is None and getattr(last_physical_attempt_capture(), "attempt_id", None) == attempt_ids[-1]:
        drive_root = current_physical_attempt_drive_root()
    return attempt_money_fact(drive_root, attempt_ids[-1]) if drive_root is not None else {}


def _emit_simple_usage(ctx: ToolContext, *, provider: str, model: str, api_key_type: str = "",
                       usage: Dict[str, Any] | None = None, settled: Dict[str, Any] | None = None) -> None:
    """One ``llm_usage`` display event of a paid leg. Its money and its four token counts are ``settled``, what the
    attempt ledger holds for the reply's own attempt (``_usage_settlement.attempt_money_fact``): the amount, final
    or the ledger's own estimate, whether the row found the bill unreadable, and the counts as that row normalized
    them. Nothing is priced, judged or counted again here, and the reply's own ``cost_invalid`` label is not the
    row's; without that fact the event states no amount, no validity and no count."""
    if not hasattr(ctx, "pending_events"):
        return
    metadata = getattr(ctx, "task_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    try:
        from ouroboros.event_usage import REPORTED_TOKENS_KEY, TOKEN_COUNTER_ALIASES

        settled = settled or {}
        money = {"cost": settled.get("cost"),
                 **{key: bool(settled.get(key)) for key in ("cost_final", "cost_estimated", "cost_invalid")}}
        counts = {name: settled.get(name) for name in TOKEN_COUNTER_ALIASES}  # None: the row does not know it
        ctx.pending_events.append({
            "type": "llm_usage",
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            "root_task_id": str(metadata.get("root_task_id") or ""),
            "parent_task_id": str(metadata.get("parent_task_id") or ""),
            "delegation_role": str(metadata.get("delegation_role") or ""),
            "provider": provider,
            "model": model,
            "api_key_type": api_key_type or provider,
            "model_category": "websearch",
            **counts,
            # The supervisor reads this copy first: the row's amount and counts in both places.
            "usage": {**(usage or {}), **money, REPORTED_TOKENS_KEY: counts},
            **money,
            "source": "web_search",
            "ts": utc_now_iso(),
            "category": "task",
            "accounting_authority": "physical_attempt_ledger",
            "ledger_attempt_ids": list((usage or {}).get("ledger_attempt_ids") or []),
        })
    except Exception:
        log.debug("Failed to emit web_search usage event", exc_info=True)


def _web_search_openrouter(ctx: ToolContext, query: str, model: str = "", search_context_size: str = "",
                           processing_preference: Optional[str] = None, connection: str = "") -> str:
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    try:
        active_model = _leg_model("openrouter", model)
    except Exception as exc:
        raise _wrapped_provider_error("OpenRouter", exc) from exc
    # Read once: the same resolved target gates, sends and bills the leg.
    target = _search_target("openrouter", connection, active_model)
    if not target:
        raise RuntimeError(f"OpenRouter connection {connection} is not configured" if connection
                           else "OPENROUTER_API_KEY is not configured")
    response = None
    try:
        from ouroboros.llm import LLMClient, openrouter_web_search_server_tool

        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Trying OpenRouter web search: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        with capture_attempt_ids() as attempt_ids:
            response = openrouter_web_search_server_tool(
                api_key=target["api_key"],
                target=target,
                model=active_model,
                query=query,
                search_context_size=search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE,
                accounting_scope=_accounting_scope(ctx, "web_search.openrouter"),
                timeout=_web_search_transport_timeout(ctx),
                processing_preference=processing_preference,
                **_web_search_caller_deadline(ctx),
            )
        from ouroboros._usage_response import failed_chat_completion, usage_from_response
        from ouroboros.llm_attempt import attach_processing_receipt

        body = search_api.plain(response)
        usage, cost, final = usage_from_response(response, payload=body, provider="openrouter", provider_reported_cost_only=True)
        attach_processing_receipt(target, usage, attempt_ids=attempt_ids)
        usage["ledger_attempt_ids"] = list(attempt_ids)
        _emit_simple_usage(ctx, provider="openrouter", model=str(target.get("usage_model") or active_model),
                           usage=usage, settled=_settled_fact(attempt_ids))
        error = LLMClient._provider_body_error(body)
        if error is not None:
            if (typed_search_refusal(error.get("status_code", error.get("code")), error.get("code"), error.get("type"))
                    and not refusal_conflicts(body, reported_cost=cost)):
                return json.dumps({"error": sanitize_tool_result_for_log(str(error.get("message") or "Search refused"))[:500],
                                   "ledger_attempt_ids": attempt_ids})
            return _unknown_provider_outcome("openrouter_server_tool", attempt_ids)
        if failed_chat_completion(body):
            return _unknown_provider_outcome("openrouter_server_tool", attempt_ids)
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        text = str((message or {}).get("content") or "").strip()
        return json.dumps({
            "answer": text or "(no answer)",
            "answer_type": "summary",
            "sources": search_api.cited_sources(body),
            "backend": "openrouter_server_tool",
            "model": active_model, "reported_model": getattr(response, "model", None), "ledger_attempt_ids": list(attempt_ids),
        }, ensure_ascii=False, indent=2)
    except UsageAccountingError as exc:
        if _own_window_refused(ctx, exc):
            return _web_search_deadline_result()
        raise
    except Exception as exc:
        if response is not None:
            return _unusable_reply("openrouter_server_tool", exc, attempt_ids)
        raise _wrapped_provider_error("OpenRouter", exc) from exc


def _web_search_anthropic(ctx: ToolContext, query: str, model: str = "",
                          processing_preference: Optional[str] = None, connection: str = "") -> str:
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    active_model = _leg_model("anthropic", model)
    target = _search_target("anthropic", connection, active_model)
    if not target:
        raise RuntimeError(f"Anthropic connection {connection} is not configured" if connection
                           else "ANTHROPIC_API_KEY is not configured")
    response = None
    try:
        import anthropic  # noqa: F401 — the SDK's own import failure precedes any attempt claim
        from ouroboros.llm import anthropic_web_search_server_tool

        if getattr(ctx, "emit_progress_fn", None):
            ctx.emit_progress_fn(f"🔍 Trying Anthropic web search: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        with capture_attempt_ids() as attempt_ids:
            response = anthropic_web_search_server_tool(
                api_key=target["api_key"], target=target,
                model=active_model,
                query=query,
                accounting_scope=_accounting_scope(ctx, "web_search.anthropic"),
                timeout=_web_search_transport_timeout(ctx),
                processing_preference=processing_preference,
                **_web_search_caller_deadline(ctx),
            )
        blocks = search_api.plain(getattr(response, "content", []) or [])
        text_parts: list[str] = []
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict) and str(block.get("type") or "") == "text":
                    text_parts.append(str(block.get("text") or ""))
        usage = search_api.plain(getattr(response, "usage", None))
        if not isinstance(usage, dict):
            usage = {}
        from ouroboros.llm_attempt import attach_processing_receipt

        attach_processing_receipt(target, usage, attempt_ids=attempt_ids)
        usage["ledger_attempt_ids"] = list(attempt_ids)
        from ouroboros.event_usage import REPORTED_TOKENS_KEY, anthropic_token_counts

        usage[REPORTED_TOKENS_KEY] = anthropic_token_counts(usage)  # native input_tokens is not the total (#1578)
        _emit_simple_usage(ctx, provider="anthropic", model=target["usage_model"], usage=usage,
                           settled=_settled_fact(attempt_ids))
        return json.dumps({
            "answer": "".join(text_parts).strip() or "(no answer)",
            "answer_type": "summary",
            "sources": search_api.cited_sources(response),
            "backend": "anthropic_server_tool",
            "model": active_model, "reported_model": getattr(response, "model", None), "ledger_attempt_ids": list(attempt_ids),
        }, ensure_ascii=False, indent=2)
    except UsageAccountingError as exc:
        if _own_window_refused(ctx, exc):
            return _web_search_deadline_result()
        raise
    except Exception as exc:
        if response is not None:
            return _unusable_reply("anthropic_server_tool", exc, attempt_ids)
        raise _wrapped_provider_error("Anthropic", exc) from exc


def _unusable_reply(backend, error, attempt_ids):
    return json.dumps({"backend": backend, "error": sanitize_tool_result_for_log(str(error))[:500],
                       "reason_code": "provider_response_failed", "ledger_attempt_ids": attempt_ids})


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


def _web_search_deepseek(ctx: ToolContext, query: str, connection: str = "") -> str:
    """DeepSeek's own Search API (``gateways/search_api.py``), an explicit Source only: the query as written in
    one physical attempt priced from its pinned quote (``_usage_settlement``), else unknown, never read as free;
    an ambiguous send is never repeated."""
    from ouroboros._usage_wait import close_after_model_send
    from ouroboros.deadline_utils import physical_dispatch_timeout
    from ouroboros.llm_attempt import _PhysicalSendNotStarted
    from ouroboros.net_transport import web_search_openai_client
    from ouroboros.provider_models import DEEPSEEK_BASE_URL

    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    refusal = search_api.query_refusal(query)
    if refusal:  # it cannot be sent as written: nothing is reserved or sent
        return json.dumps({**refusal, "backend": "deepseek_search_tool"}, ensure_ascii=False, indent=2)
    api_key = _search_key("deepseek", connection)
    if not api_key:
        raise RuntimeError(f"DeepSeek connection {connection} is not configured" if connection
                           else "DEEPSEEK_API_KEY is not configured")
    scope = _accounting_scope(ctx, "web_search.deepseek")
    from ouroboros.pricing_catalog import pin_tariff

    # This key's per-request quote and currency, pinned before the send; settlement prices the answer it got.
    pinned = pin_tariff({"provider": "deepseek", "base_url": DEEPSEEK_BASE_URL, "api_key": api_key}, search=True,
                        drive_root=scope.drive_root)
    request = AttemptRequest(model="", provider="deepseek", force_unknown_reservation=True, drive_root=scope.drive_root,
                             task_id=scope.task_id, root_task_id=scope.root_task_id, parent_task_id=scope.parent_task_id,
                             category=scope.category, source=scope.source, cost_evidence=pinned)
    _search_progress(ctx, f"🔍 Trying DeepSeek web search: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
    client = web_search_openai_client(api_key=api_key, base_url=DEEPSEEK_BASE_URL,
                                      timeout=_web_search_transport_timeout(ctx), follow_redirects=False)

    def send():
        timeout = physical_dispatch_timeout(_web_search_transport_timeout(ctx))
        if _web_search_deadline_exhausted(ctx):  # this leg's window, spent after its claim: positively unsent
            raise _PhysicalSendNotStarted("deadline")
        return search_api.send_search(client, query, timeout=timeout)

    try:
        with capture_attempt_ids() as attempt_ids:
            try:
                payload = execute_physical_attempt(request, send, extractor=search_api.search_usage)
            finally:
                close_after_model_send(client)  # a sender that outlives its receiver keeps it until its send ends
    except UsageAccountingError as exc:
        if _own_window_refused(ctx, exc):
            return _web_search_deadline_result()
        raise
    except Exception as exc:
        raise _wrapped_provider_error("DeepSeek", exc) from exc
    return json.dumps({**search_api.search_result(payload), "backend": "deepseek_search_tool",
                       "ledger_attempt_ids": list(attempt_ids)}, ensure_ascii=False, indent=2)


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


def _stream_failure_error(event_type: str, error: dict, response: dict) -> RuntimeError:
    """Retain the event's native error fields, including any reported HTTP status."""
    detail = sanitize_tool_result_for_log(str(error.get("message") or "no detail"))[:300]
    exc = RuntimeError(f"OpenAI web search {event_type}: {detail}")
    exc.body = response
    for obj in (error, response):
        for field in ("status_code", "http_status", "status", "code"):
            value = obj.get(field)
            try:
                status = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= status <= 599:
                setattr(exc, "status_code", status)
                return exc
    return exc


def _responses_search_candidate(ctx, query: str, resolved: dict, options: dict):
    """One complete Responses body supplies capture, reservation and actual send."""
    from ouroboros.llm_attempt import (
        _attempt_request, _candidate_before_dispatch, _finalized_physical_candidate,
        apply_processing_preference,
    )
    target = {**resolved, "api_surface": "responses",
              "processing_preference": options["preference"], "requested_reasoning_effort": options["effort"]}
    payload = {"model": target["resolved_model"],
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
    connection: str = "",
    _resolved_target: Optional[dict] = None,
) -> str:
    from ouroboros.model_slots import resolve_processing_preference

    preference = resolve_processing_preference("websearch", override=_processing_preference)
    if _web_search_deadline_exhausted(ctx):
        return _web_search_deadline_result()
    def failed(message, **facts):
        return json.dumps({"error": message, "backend": "openai_responses", "ledger_attempt_ids": attempt_ids, **facts})
    attempt_ids = []
    active_model = model or runtime_setting("OUROBOROS_WEBSEARCH_MODEL", DEFAULT_SEARCH_MODEL)
    resolved = _resolved_target or _search_target("openai", connection, active_model)
    provider = str(resolved.get("provider") or "openai") if resolved else "openai"
    if not resolved or not str(resolved.get("api_key") or "").strip():
        return failed("Official OpenAI search is unavailable")
    active_context = search_context_size or DEFAULT_SEARCH_CONTEXT_SIZE
    active_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    reservation = None
    dispatched = False
    send_started = False
    stream = None
    refused_before_work = work_seen = False
    terminal = False
    terminal_response = None
    failure_usage = None
    manifest = None
    try:
        from ouroboros._usage_response import processing_receipt, search_failure_usage, usage_from_response
        from ouroboros.config import get_finalization_grace_sec
        from ouroboros.deadline_utils import physical_dispatch_timeout
        from ouroboros.llm_attempt import _PhysicalSendNotStarted
        from ouroboros.net_transport import web_search_openai_client
        transport_timeout = _web_search_transport_timeout(ctx)
        # Explicit per-attempt transport bound; the outer ToolEntry covers the
        # configured paid cascade rather than acting as the socket timeout.
        client = web_search_openai_client(
            api_key=resolved["api_key"], base_url=resolved["base_url"],
            default_headers=resolved.get("default_headers"),
            timeout=transport_timeout,
        )
        target, candidate, request, before_dispatch = _responses_search_candidate(ctx, query, resolved, {
            "preference": preference, "submission": _processing_submission,
            "search_context_size": active_context, "effort": active_effort})
        reservation = reserve_attempt(request)
        attempt_ids.append(reservation.attempt_id)
        manifest = before_dispatch(reservation)
        if _web_search_deadline_exhausted(ctx):  # after recording, which spends the same clock (#572)
            release_attempt(reservation, "deadline_exhausted_before_dispatch", candidate_manifest_ref=manifest)
            reservation = None
            return _web_search_deadline_result()
        # This leg reads its own stream: its process is recorded as the sender (its death is provable),
        # never the task's model wait, whose receiver stays unbound (``usage_accounting.mark_dispatched``).
        mark_dispatched(reservation, candidate_manifest_ref=manifest, local_answer_owner_pid=os.getpid(),
                        task_receiver=False)
        dispatched = True
        if _attempt == 0:
            # An attempt, not evidence that a search ran: that is the stream's own web_search_call below.
            _search_progress(ctx, f"🔍 Trying OpenAI web search: {sanitize_tool_result_for_log(str(query or ''))[:100]}")
        # At the send, after the claim and its progress row: inherited controls and windows narrow every
        # socket phase, then this leg's deadline less its reserve admits the bytes. Either refusal is
        # positively unsent, so the claim is released, never left unknown (#572).
        timeout = physical_dispatch_timeout(_web_search_transport_timeout(ctx))
        if _web_search_deadline_exhausted(ctx):
            raise _PhysicalSendNotStarted("deadline")
        try:
            send_started = True
            stream = client.responses.create(**candidate, timeout=timeout)
        except Exception as error:
            from ouroboros.llm_attempt import processing_refusal

            refusal = processing_refusal(target, candidate, error)
            if refusal is not error:
                raise refusal from error
            if is_processing_refusal(error):
                raise
            terminal_response = refusal_failure_payload(error)
            work_seen = refusal_conflicts(error)
            if work_seen:
                raise
            native = terminal_response.get("error", terminal_response) if isinstance(terminal_response, dict) else {}
            if isinstance(native, dict):
                refused_before_work = typed_search_refusal(_provider_status_code(error), native.get("code"), native.get("type"))
            raise
        text_parts: list[str] = []
        usage: dict = {}
        sources: List[Dict[str, str]] = []
        progress_sent = False
        reported_cost, reported_cost_final = None, False

        for event in stream:
            etype = getattr(event, "type", "")

            if etype in ("response.failed", "error", "response.incomplete"):
                resp_obj = getattr(event, "response", None)
                received = resp_obj if resp_obj is not None else event
                body = search_api.plain(received)
                terminal_response = body if isinstance(body, dict) else {}
                terminal = etype != "error" and resp_obj is not None and isinstance(body, dict)
                err = terminal_response.get("error") or search_api.plain(getattr(event, "error", None)) or terminal_response
                native = err if isinstance(err, dict) else {}
                failure = _stream_failure_error(etype, native, terminal_response)
                failure_usage = usage_from_response(received, payload=terminal_response, provider="openai", provider_reported_cost_only=True)
                if (terminal and etype == "response.failed" and not work_seen
                        and not refusal_conflicts(failure, reported_cost=failure_usage[1])):
                    refused_before_work = typed_search_refusal(_provider_status_code(failure), native.get("code"), native.get("type"))
                raise failure

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
                terminal = resp_obj is not None
                if terminal:
                    terminal_response = resp_obj
                    usage, reported_cost, reported_cost_final = usage_from_response(resp_obj)
                    receipt = processing_receipt("openai", usage,
                        requested=request.processing_preference, submitted_native=request.submitted_processing_mode)
                    if receipt is not None:
                        usage["processing"] = receipt
                    sources = search_api.cited_sources(resp_obj)
                break
            if etype not in ("response.created", "response.queued", "response.in_progress"):
                work_seen = True

        text = "".join(text_parts)
        usage["ledger_attempt_ids"] = [reservation.attempt_id]
        try:
            if terminal:
                settle_attempt(reservation, usage, cost_usd=reported_cost,
                               cost_final=reported_cost_final)
            else:
                from ouroboros.llm_stream import IncompleteProviderStream
                from ouroboros.usage_accounting import _retain_physical_failure

                # The read ended without terminal framing: this receive path is over,
                # its outcome unknown — the same retained fact a raising send leaves.
                incomplete = IncompleteProviderStream("Responses stream ended without response.completed")
                _retain_physical_failure(reservation, exc=incomplete)
                mark_unresolved(reservation, str(incomplete))
            dispatched = False
        except Exception as exc:
            # Preserve an already-paid result; unresolved retains its upper bound.
            log.exception("Failed to settle OpenAI Responses search attempt")
            try:
                mark_unresolved(reservation, f"settlement_write_failed:{type(exc).__name__}")
                dispatched = False
            except Exception:
                log.exception("Failed to mark Responses settlement unresolved")

        if not terminal:
            return _unknown_provider_outcome("openai_responses", attempt_ids)

        _emit_simple_usage(ctx, provider=provider, model=reservation.model, usage=usage,
                           settled=_settled_fact([reservation.attempt_id], reservation.drive_root))

        if text.strip() or sources:
            return json.dumps({"answer": text or "(no answer)", "answer_type": "summary", "sources": sources, "backend": "openai_responses", "model": active_model, "reported_model": getattr(resp_obj, "model", None), "ledger_attempt_ids": attempt_ids}, ensure_ascii=False, indent=2)
        # A complete but empty result can advance Auto; its attempt and settlement stay here.
        return failed("OpenAI web search returned no answer and no sources")
    except Exception as e:
        proven_not_started = False
        processing_refused = False
        if dispatched and stream is None and is_processing_refusal(e):
            from ouroboros.usage_accounting import _terminalize_failed_attempt

            try:
                state = _terminalize_failed_attempt(reservation, e)
            except Exception:
                log.exception("Failed to record the processing refusal")
                return _unknown_provider_outcome("openai_responses", attempt_ids)
            if state != "settled" or getattr(e, "processing_refusal_conflict", False):
                return _unknown_provider_outcome("openai_responses", attempt_ids)
            dispatched = False
            processing_refused = refused_before_work = True
        elif dispatched and stream is None and not work_seen:
            proven_not_started = release_pre_dispatch_attempt(reservation, e)
            dispatched = not proven_not_started
        was_dispatched = send_started and not proven_not_started
        if reservation is not None and dispatched:
            from ouroboros.usage_accounting import _retain_physical_failure

            if terminal_response is None:  # SDK iterators can raise with a received error envelope.
                terminal_response = refusal_failure_payload(e)
            _retain_physical_failure(reservation, exc=e)  # this send's receive path ended raising
            try:
                usage, cost, final = search_failure_usage(e, terminal_response, failure_usage, work_seen=work_seen)
                if terminal or refused_before_work or cost is not None:
                    settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
                else:
                    mark_unresolved(reservation, f"{type(e).__name__}: {e}")
            except Exception:
                log.exception("Failed to record OpenAI Responses search failure")
        if reservation is not None and (not send_started or proven_not_started):
            from ouroboros.usage_accounting import _pre_dispatch_failure

            failure = _pre_dispatch_failure(reservation, request, e, candidate_manifest_ref=manifest)
            if _own_window_refused(ctx, failure):
                return _web_search_deadline_result()
            if isinstance(e, UsageAccountingError) or failure.physical_attempt_capture.state != "released":
                raise failure from e
        if isinstance(e, UsageAccountingError):
            if reservation is None and _own_window_refused(ctx, e):  # refused before anything was reserved
                return _web_search_deadline_result()
            raise
        detail = sanitize_tool_result_for_log(str(e))[:500]
        status = _provider_status_code(e)
        if was_dispatched and not refused_before_work:
            if terminal:
                return failed(detail, reason_code="provider_response_failed")
            return _unknown_provider_outcome("openai_responses", attempt_ids)
        retryable_refusal = refused_before_work and status in (429, 503, 529)
        if _attempt == 0 and (
            (_is_timeout_error(e) and not was_dispatched) or retryable_refusal or processing_refused
        ):
            from ouroboros.deadline_utils import deadline_remaining_sec, has_deadline

            if has_deadline(ctx) and deadline_remaining_sec(ctx) <= max(
                1.0, float(get_finalization_grace_sec())
            ):
                return failed(f"OpenAI web search timed out before a safe retry: {detail}")
            return _web_search_openai(
                ctx, query, model=active_model, search_context_size=search_context_size,
                reasoning_effort=reasoning_effort, _attempt=1,
                _processing_preference=preference,
                _resolved_target=resolved,
                _processing_submission="standard" if processing_refused else _processing_submission,
                **({"connection": connection} if connection else {}),
            )
        return failed(f"OpenAI web search failed ({type(e).__name__}): {detail}")


def _web_search(ctx: ToolContext, query: str, model: str = "", search_context_size: str = "",
                reasoning_effort: str = "") -> str:
    from ouroboros.model_slots import resolve_processing_preference

    # A model this call names is mandatory (per_call); an empty one is an ordinary omission.
    call_model = str(model or "").strip()
    route = resolve_web_search_route(model=call_model or None, per_call=bool(call_model))
    tried = []
    preference = resolve_processing_preference("websearch")
    unavailable = (f"web_search unavailable: no configured built-in route can send {call_model}, the model this call "
                   "named (route.unavailable says why); omit model to use the configured Source and Model."
                   if call_model else f"web_search unavailable: source {route['source']} has no configured search backend "
                   "that succeeded. Configure a search API key or install optional ddgs.")
    result = {"error": route.get("error") or unavailable, "backend": route["source"]}
    # Auto tries eligible routes; a pinned source always reaches its own backend,
    # which reports its own missing credential or accounting refusal.
    for leg in route["legs"] + ([] if route["source"] == "auto" else route["unavailable"]):
        if _web_search_deadline_exhausted(ctx):
            result = json.loads(_web_search_deadline_result())
            break
        if tried:  # only a typed failure reaches another leg; say so before it is tried
            _search_progress(ctx, f"↪ Trying {_SOURCE_LABELS[leg['source']]} after {_SOURCE_LABELS[tried[-1]['source']]} "
                                  "web search failed; details in Logs")
        try:
            with capture_attempt_ids() as ids:
                # The selection's named key travels with its leg; other legs use the built-in key.
                named = {"connection": leg["connection"]} if leg.get("connection") else {}
                if leg["source"] == "openai":
                    raw = _web_search_openai(ctx, query, model=leg["model"], search_context_size=search_context_size,
                                             reasoning_effort=reasoning_effort, _processing_preference=preference, **named)
                elif leg["source"] == "openrouter":
                    raw = _web_search_openrouter(ctx, query, model=leg["model"], search_context_size=search_context_size,
                                                 processing_preference=preference, **named)
                elif leg["source"] == "anthropic":
                    raw = _web_search_anthropic(ctx, query, model="anthropic::" + leg["model"], processing_preference=preference,
                                                **named)
                elif leg["source"] == "deepseek":
                    raw = _web_search_deepseek(ctx, query, **named)
                else:
                    raw = _web_search_ddgs(query)
            result = json.loads(raw)
        except UsageAccountingError:
            raise
        except Exception as exc:
            result = (json.loads(_unknown_provider_outcome(leg["backend"])) if _provider_outcome_is_unknown(exc)
                      else {"error": sanitize_tool_result_for_log(str(exc))[:500]})
            capture = _capture_on_chain(exc)
            if capture and getattr(capture, "attempt_id", None):
                ids.append(capture.attempt_id)
            ids.extend(getattr(exc, "ledger_attempt_ids", []) or [])
        # A pinned source's failure is reported as that source; legs_tried names the backend.
        pinned_failure = route["source"] != "auto" and "error" in result and result.get("reason_code") is None
        result.update(backend=route["source"] if pinned_failure else leg["backend"], model=leg["model"])
        tried.append({**leg, "outcome": result.get("reason_code") or ("failed" if "error" in result else "succeeded"),
                      "ledger_attempt_ids": list(dict.fromkeys([*ids, *result.get("ledger_attempt_ids", [])])),
                      **({"error": result["error"]} if "error" in result else {})})
        if "error" not in result or result.get("reason_code") in {"provider_outcome_unknown", "provider_response_failed", "deadline_exhausted"}:
            break
    last = _SOURCE_LABELS.get(tried[-1]["source"], "") if tried else ""
    if "error" not in result and len(tried) > 1:
        # A summary names who answered; it is no proof that a provider search ran or was billed.
        _search_progress(ctx, f"🔍 Answer from {last} after {len(tried) - 1} failed source{'s' if len(tried) > 2 else ''}")
    elif result.get("reason_code") == "provider_outcome_unknown":
        _search_progress(ctx, f"⚠️ Web search outcome unknown ({last}); no further attempt or fallback was sent")
    elif result.get("reason_code") == "deadline_exhausted":
        _search_progress(ctx, "⚠️ Web search stopped: the owner deadline leaves no dispatch window")
    elif result.get("reason_code") in {"query_limit", "argument_invalid"}:  # refused here; nothing was sent
        _search_progress(ctx, f"⚠️ Web search not sent: {last} does not accept this query as written")
    elif "error" in result and tried:
        # The exception text stays in the result and Logs, never in the owner's row.
        _search_progress(ctx, f"⚠️ Web search failed ({last}); no source answered; details in Logs")
    # An omitted call never tries a named connection; with nothing else eligible, say which one a call may name.
    named = ([f"{leg['source']}@{leg['connection']}" for connection in named_connection_routes()
              for leg in connection["legs"]] if "error" in result and not call_model and not route["legs"] else [])
    text = json.dumps({**result, "route": route, "legs_tried": tried,
                       "backend_errors": [leg["error"] for leg in tried if "error" in leg],
                       **({"named_connections": named} if named else {})}, ensure_ascii=False, indent=2)
    if "error" in result:
        from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

        # The cascade owns this failure object. An unknown provider outcome
        # keeps its ledger/custody and never acquires a completion fact here.
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=text))
    return text


_SOURCE_LABELS = {"openai": "OpenAI", "openrouter": "OpenRouter", "anthropic": "Anthropic", "ddgs": "DDGS",
                  "deepseek": "DeepSeek"}


def _search_progress(ctx: ToolContext, text: str) -> None:
    if getattr(ctx, "emit_progress_fn", None):
        ctx.emit_progress_fn(text)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                'Search with the configured Source and Model (a blank Model is the recommended one). Returns a summary '
                'lead with sources; open them with browse_page to verify load-bearing facts. Auto uses eligible built-in '
                'routes; a named source is strict. A model you pass is mandatory: only it is sent (under Auto an id without '
                'source:: may also go through another provider of that model), and if it cannot run the call fails and says why. '
                'Split compound questions into focused searches. Skills, MCP and browser tools remain separate choices.'
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "A single focused search query (split compound asks into separate calls)."},
                "model": {"type": "string", "description": "This call's model: native id or routed source[@connection]::model; must agree with a pinned Source (ddgs and DeepSeek take none). Omit for the configured one."},
                "search_context_size": {"type": "string", "enum": ["low", "medium", "high"],
                                        "description": f"How much context to fetch (default: {DEFAULT_SEARCH_CONTEXT_SIZE})"},
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"],
                                     "description": f"Reasoning effort (default: {DEFAULT_REASONING_EFFORT})"},
            }, "required": ["query"]},
        }, _web_search, timeout_sec=_web_search_outer_timeout_sec()),
    ]
