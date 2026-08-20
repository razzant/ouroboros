"""One-shot, physically accounted LLM capability probes.

Probe transports deliberately do not use the ordinary chat retry, fallback,
reasoning, cache, or capability-learning paths.  Target resolution and client
construction remain owned by the routing leaf the client composes
(:mod:`ouroboros.llm_routing`); this module only builds the final probe candidate
and dispatches it through the existing physical-attempt accounting seam.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ouroboros.usage_accounting import (
    UsageScope,
    current_usage_scope,
    physical_attempt_limit,
    usage_scope,
)

PROVIDER_TEST_MAX_TOKENS = 16
PROVIDER_TEST_PROMPT = "Reply OK"


class ProbeEnvelopeError(RuntimeError):
    """A provider returned HTTP success without a completion envelope."""


def _accounted_send(
    target: dict[str, Any],
    candidate: dict[str, Any],
    send,
    *,
    source: str,
) -> Any:
    # Named at the owner leaf, not through the llm.py facade: an llm_* leaf never
    # imports its parent (the composition would cycle), and a test that patches the
    # executor patches llm_attempt, which is where `_execute_candidate` reads it.
    # The import stays lazy so importing this module costs nothing at startup.
    from ouroboros.llm_attempt import (
        _attempt_request,
        _candidate_before_dispatch,
        _execute_candidate,
        _physical_candidate,
    )

    final_candidate = _physical_candidate(candidate)
    request = _attempt_request(target, final_candidate, source=source)
    return _execute_candidate(
        request,
        lambda: send(final_candidate),
        _candidate_before_dispatch(final_candidate, request),
    )


def probe_oversized_context(
    client,
    model: str,
    content: str,
    *,
    base_url: str = "",
    max_output_tokens: int = 8,
    timeout: float = 20.0,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Compatibility implementation for the generative context-window probe."""
    try:
        target = client._resolve_remote_target(model)
        if str(base_url or "").strip():
            target = {**target, "base_url": str(base_url).strip()}
        if api_key is not None:
            target = {**target, "api_key": api_key}
        remote = client._get_remote_client(target)
        resolved_model = str(target.get("resolved_model") or model.split("::")[-1])
        provider = str(target.get("provider") or "")
    except Exception as exc:  # pragma: no cover - setup failure -> fail-closed
        return {
            "ok": False,
            "status_code": None,
            "body": f"probe setup failed: {type(exc).__name__}",
            "echoed_text": "",
            "usage_prompt": 0,
        }

    cap = (
        {"max_completion_tokens": max_output_tokens}
        if provider == "openai"
        else {"max_tokens": max_output_tokens}
    )
    candidate = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        **cap,
    }

    def dispatch() -> Any:
        return _accounted_send(
            target,
            candidate,
            lambda payload: remote.with_options(timeout=timeout).chat.completions.create(**payload),
            source="capability_probe",
        )

    try:
        if current_usage_scope() is None:
            with usage_scope(UsageScope(
                task_id="system:capability_probe",
                root_task_id="system:capability_probe",
                category="capability_probe",
                source="capability_probe",
            )):
                response = dispatch()
        else:
            response = dispatch()
        echoed, usage_prompt = "", 0
        try:
            echoed = str(response.choices[0].message.content or "")
            usage_prompt = int(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0)
        except Exception:
            pass
        return {
            "ok": True,
            "status_code": 200,
            "body": "",
            "echoed_text": echoed,
            "usage_prompt": usage_prompt,
        }
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        body = str(getattr(exc, "message", "") or getattr(exc, "body", "") or str(exc))
        return {
            "ok": False,
            "status_code": status if isinstance(status, int) else None,
            "body": body,
            "echoed_text": "",
            "usage_prompt": 0,
        }


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    json_method = getattr(value, "json", None)
    if callable(json_method):
        try:
            parsed = json_method()
            return parsed if not isinstance(parsed, str) else value
        except Exception:
            pass
    try:
        return vars(value)
    except TypeError:
        return value


def _valid_completion_envelope(response: Any, provider: str) -> bool:
    payload = _plain(response)
    if not isinstance(payload, dict):
        return False
    if provider == "anthropic":
        return isinstance(payload.get("content"), list)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = _plain(choices[0])
    if not isinstance(first, dict) or "message" not in first:
        return False
    return isinstance(_plain(first.get("message")), dict)


def _error_facts(exc: BaseException) -> tuple[Optional[int], str, str]:
    capture = getattr(exc, "physical_attempt_capture", None)
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status is None and capture is not None:
        status = getattr(capture, "provider_status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError, OverflowError):
        status = None
    code = str(getattr(exc, "code", "") or "").strip().lower()
    error_type = str(getattr(exc, "type", "") or "").strip().lower()
    if capture is not None:
        code = str(getattr(capture, "provider_code", "") or code).strip().lower()
        error_type = str(
            getattr(capture, "provider_error_type", "") or error_type
        ).strip().lower()
    return status, code, error_type


def controlled_probe_error(exc: BaseException) -> dict[str, Any]:
    """Map typed transport facts to one bounded, provider-neutral reason."""
    status, code, error_type = _error_facts(exc)
    credit_codes = {
        "billing_hard_limit_reached",
        "credit_balance_too_low",
        "credits_exhausted",
        "insufficient_credits",
        "insufficient_quota",
    }
    model_codes = {"model_not_found", "unknown_model"}

    if status == 402 or code in credit_codes or error_type in credit_codes:
        reason = "No credits"
    elif status == 401:
        reason = "Invalid key"
    elif status == 403:
        reason = "Access denied"
    elif status == 404 or code in model_codes or error_type in model_codes:
        reason = "Model unavailable"
    elif status == 429:
        reason = "Rate limited"
    elif status is not None and 500 <= status < 600:
        reason = "Provider unavailable"
    else:
        timeout_types: tuple[type[BaseException], ...] = (TimeoutError,)
        connect_types: tuple[type[BaseException], ...] = (ConnectionError,)
        try:
            import httpx

            timeout_types += (httpx.TimeoutException,)
            connect_types += (httpx.ConnectError, httpx.TransportError)
        except Exception:  # pragma: no cover - dependency is shipped
            pass
        try:
            import requests

            timeout_types += (requests.Timeout,)
            connect_types += (requests.ConnectionError,)
        except Exception:  # pragma: no cover - dependency is shipped
            pass
        try:
            import openai

            timeout_types += (openai.APITimeoutError,)
            connect_types += (openai.APIConnectionError,)
        except Exception:  # pragma: no cover - dependency is shipped
            pass
        if isinstance(exc, timeout_types):
            reason = "Timed out"
        elif isinstance(exc, connect_types):
            reason = "Could not reach provider"
        else:
            reason = "Model request failed"
    return {
        "ok": False,
        "error": reason,
        "status_code": status,
        "exception_type": type(exc).__name__,
    }


def _probe_candidate(target: Mapping[str, Any]) -> dict[str, Any]:
    provider = str(target.get("provider") or "")
    model = str(target.get("resolved_model") or "")
    token_key = (
        "max_completion_tokens"
        if provider == "openai" and model.startswith(("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    candidate: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROVIDER_TEST_PROMPT}],
        token_key: PROVIDER_TEST_MAX_TOKENS,
    }
    if provider == "openrouter":
        candidate["extra_body"] = {"provider": {"allow_fallbacks": False}}
    return candidate


def _target_is_configured(target: Mapping[str, Any]) -> bool:
    provider = str(target.get("provider") or "")
    if provider == "openai-compatible":
        return bool(str(target.get("base_url") or "").strip())
    if provider == "gigachat":
        return bool(
            str(target.get("api_key") or "").strip()
            or (
                str(target.get("user") or "").strip()
                and str(target.get("password") or "").strip()
            )
        )
    return bool(str(target.get("api_key") or "").strip())


def probe_provider_readiness(
    client,
    model: str,
    *,
    settings: Mapping[str, Any],
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Send one short request through one request-local provider target."""
    target: dict[str, Any] = {}
    remote_client = None
    try:
        target = client._resolve_remote_target(model, settings=settings)
        if not _target_is_configured(target):
            return {
                "ok": False,
                "error": "Provider is not configured",
                "status_code": None,
                "exception_type": "ProviderNotConfigured",
            }
        candidate = _probe_candidate(target)
        provider = str(target.get("provider") or "")

        if provider in {
            "openrouter", "openai", "openai-compatible", "minimax", "cloudru",
        }:
            remote_client = client._new_remote_client(target)

            def send_openai(payload):
                scoped = remote_client.with_options(timeout=timeout)
                return scoped.chat.completions.create(**payload)

            dispatch = send_openai

        elif provider == "anthropic":
            import requests

            url = f"{str(target.get('base_url') or '').rstrip('/')}/messages"
            headers = {
                "x-api-key": str(target.get("api_key") or ""),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            def send_anthropic(payload):
                response = requests.post(
                    url, headers=headers, json=payload, timeout=float(timeout),
                )
                response.raise_for_status()
                return response

            dispatch = send_anthropic

        elif provider == "gigachat":
            # The GigaChat library otherwise imports GIGACHAT_ACCESS_TOKEN from
            # the environment.  A stale inherited token makes its auth wrapper
            # resend once after 401 even when transport retries are disabled.
            remote_client = client._new_gigachat_client(
                {**target, "access_token": ""}, timeout=timeout, max_retries=0,
            )

            def send_gigachat(payload):
                return remote_client.chat(payload)

            dispatch = send_gigachat

        else:
            raise ValueError("unsupported provider route")

        with usage_scope(UsageScope(
            task_id="system:provider_test",
            root_task_id="system:provider_test",
            category="provider_test",
            source="provider_test",
        )), physical_attempt_limit(1):
            response = _accounted_send(
                target, candidate, dispatch, source="provider_test",
            )
        if not _valid_completion_envelope(response, provider):
            raise ProbeEnvelopeError("provider returned no completion envelope")
        return {
            "ok": True,
            "status_code": 200,
            "exception_type": "",
        }
    except Exception as exc:
        return controlled_probe_error(exc)
    finally:
        close = getattr(remote_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


__all__ = [
    "PROVIDER_TEST_MAX_TOKENS",
    "PROVIDER_TEST_PROMPT",
    "controlled_probe_error",
    "probe_oversized_context",
    "probe_provider_readiness",
]
