"""Shared, secretless contracts for the trusted provider integration lane."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from enum import Enum

import pytest

from ouroboros.llm_stream import ProviderStreamError
from ouroboros.provider_models import (
    OPENAI_DIRECT_DEFAULTS,
    normalize_deepseek_reasoning_effort,
    normalize_model_identity,
)
from ouroboros.utils import sanitize_tool_result_for_log

CANARY_TIMEOUT_SEC = 120.0
CANARY_MAX_TOKENS = 1024
CANARY_CONTINUATION_MAX_TOKENS = 2048
CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS = 2
CANARY_EMPTY_RESPONSE_BACKOFF_SEC = 1.0
CANARY_TOOL_NAME = "delegate_start"
CANARY_SUBAGENT_ID = "provider-contract-canary"


@dataclass(frozen=True)
class ProviderCanary:
    """One physical provider/API surface in the trusted full-registry matrix."""

    canary_id: str
    model: str
    expected_provider: str
    credential_env: str
    credential_required: bool
    reasoning_effort: str
    named_tool_choice: bool = True
    continue_to_final: bool = False


_OPENROUTER_CANARIES = (
    ProviderCanary(
        "openrouter_gemini", "google/gemini-3.8-flash", "openrouter",
        "OPENROUTER_API_KEY", True, "medium",
    ),
    ProviderCanary(
        "openrouter_opus", "anthropic/claude-opus-5", "openrouter",
        "OPENROUTER_API_KEY", True, "medium",
    ),
    ProviderCanary(
        "openrouter_fable", "anthropic/claude-fable-5.1", "openrouter",
        "OPENROUTER_API_KEY", True, "medium", named_tool_choice=False,
    ),
    ProviderCanary(
        "openrouter_gpt", "openai/gpt-5.6-luna", "openrouter",
        "OPENROUTER_API_KEY", True, "medium",
    ),
    ProviderCanary(
        "openrouter_grok", "x-ai/grok-4.6", "openrouter",
        "OPENROUTER_API_KEY", True, "medium",
    ),
    ProviderCanary(
        "openrouter_deepseek", "deepseek/deepseek-v4-pro-0813", "openrouter",
        "OPENROUTER_API_KEY", True, "medium",
    ),
)

_DIRECT_ANTHROPIC_CANARY = ProviderCanary(
    "anthropic_direct", "anthropic::claude-sonnet-5", "anthropic",
    "ANTHROPIC_API_KEY", True, "medium",
)

_OPTIONAL_DIRECT_CANARIES = (
    ProviderCanary(
        "minimax_direct", "minimax::MiniMax-M3", "minimax",
        "MINIMAX_API_KEY", False, "none",
    ),
    ProviderCanary(
        # "medium" ON PURPOSE (unlike its optional siblings): the deepseek lane
        # CARRIES reasoning_effort, and the canary pins that carriage. The
        # continuation is mandatory too: DeepSeek's tool contract is defined by
        # replaying reasoning_content on the second request.
        "deepseek_direct", "deepseek::deepseek-v4-flash", "deepseek",
        "DEEPSEEK_API_KEY", False, "medium", continue_to_final=True,
    ),
    ProviderCanary(
        "cloudru_direct", "cloudru::zai-org/GLM-4.7", "cloudru",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY", False, "none",
    ),
    ProviderCanary(
        "gigachat_direct", "gigachat::GigaChat-2-Max", "gigachat",
        "GIGACHAT_CREDENTIALS", False, "none", named_tool_choice=False,
    ),
)


def unique_openai_direct_defaults():
    """Derive the live direct-OpenAI matrix from the shipped provider SSOT."""
    return tuple(dict.fromkeys(model for model in OPENAI_DIRECT_DEFAULTS.values() if model))


def _openai_direct_canaries():
    seen = set()
    rows = []
    main_model = OPENAI_DIRECT_DEFAULTS.get("main")
    for role, model in OPENAI_DIRECT_DEFAULTS.items():
        if not model or model in seen:
            continue
        seen.add(model)
        rows.append(ProviderCanary(
            canary_id=f"openai_direct_{role}",
            model=model,
            expected_provider="openai",
            credential_env="OPENAI_API_KEY",
            credential_required=True,
            reasoning_effort="medium",
            continue_to_final=model == main_model,
        ))
    return tuple(rows)


def provider_canary_matrix():
    """Return the exact paid matrix in stable physical-call order."""
    return (
        *_OPENROUTER_CANARIES,
        *_openai_direct_canaries(),
        _DIRECT_ANTHROPIC_CANARY,
        *_OPTIONAL_DIRECT_CANARIES,
    )


class ProviderFailureKind(str, Enum):
    RED = "red"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ProviderFailureClassification:
    kind: ProviderFailureKind
    reason: str
    status_code: int | None = None


def _exception_chain(exc: BaseException):
    chain = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return tuple(chain)


def provider_error_evidence(exc: BaseException):
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if type(status) is not int and response is not None:
        status = getattr(response, "status_code", None)
    status = status if type(status) is int else None
    body = ""
    if response is not None:
        try:
            body = str(response.text or "")
        except Exception:
            body = ""
    structured_body = getattr(exc, "body", None)
    if structured_body:
        try:
            body = "\n".join(filter(None, (body, json.dumps(structured_body))))
        except (TypeError, ValueError):
            body = "\n".join(filter(None, (body, str(structured_body))))
    chain = _exception_chain(exc)
    message = "\n".join(str(item) for item in chain)
    return status, body, message, chain


def classify_provider_failure(
    provider_id: str,
    exc: BaseException,
) -> ProviderFailureClassification:
    """Classify only explicit non-code alarms as typed inconclusive.

    Contract/auth/model/tool/reasoning 4xx stay RED. In particular, the #229
    function-tools + reasoning 400 must never be hidden by a broad text match.
    """
    del provider_id
    status, body, message, chain = provider_error_evidence(exc)
    lowered = "\n".join((body, message)).lower()
    if any(
        marker in lowered
        for marker in (
            "insufficient_quota",
            "credit balance is too low",
            "billing_hard_limit_reached",
            "billing_not_active",
            "payment_required",
            "exceeded your current quota",
            "requires more credits",
            "can only afford",
        )
    ):
        return ProviderFailureClassification(
            ProviderFailureKind.INCONCLUSIVE, "quota_or_billing", status,
        )
    if status == 429:
        return ProviderFailureClassification(
            ProviderFailureKind.INCONCLUSIVE, "rate_limit_429", status,
        )
    if status is not None and 500 <= status < 600:
        return ProviderFailureClassification(
            ProviderFailureKind.INCONCLUSIVE, "provider_5xx", status,
        )
    if status is None and any(
        isinstance(item, TimeoutError) or "timeout" in type(item).__name__.lower()
        for item in chain
    ):
        return ProviderFailureClassification(
            ProviderFailureKind.INCONCLUSIVE, "transport_timeout",
        )
    if getattr(exc, "stream_incomplete", False):
        # Main streams, so the canary's first turn does too. Weather, never
        # contract: a socket that died before the terminal frame, or the
        # provider's own SSE error frame without a 4xx code (an overload/
        # api_error shape; 429/5xx already took the ladder above). A complete
        # body the host judged unusable falls through to RED — that assembler
        # contract is what this canary guards.
        if isinstance(exc, ProviderStreamError):
            if not (status is not None and 400 <= status <= 499):  # a 4xx SSE error is contract: RED below
                return ProviderFailureClassification(
                    ProviderFailureKind.INCONCLUSIVE, "stream_provider_error", status,
                )
        elif not getattr(exc, "stream_rejected", False):
            return ProviderFailureClassification(
                ProviderFailureKind.INCONCLUSIVE, "stream_transport_loss", status,
            )
    return ProviderFailureClassification(
        ProviderFailureKind.RED, "provider_contract_or_unclassified", status,
    )


def skip_on_provider_environmental_error(
    provider_id: str,
    exc: BaseException,
) -> None:
    """Skip only classifier-approved typed inconclusive provider outcomes."""
    if isinstance(exc, AssertionError):
        return
    import sys

    classification = classify_provider_failure(provider_id, exc)
    status, body, message, _chain = provider_error_evidence(exc)
    safe_body = sanitize_tool_result_for_log(body)
    safe_message = sanitize_tool_result_for_log(message)
    if safe_body:
        print(f"[{provider_id}] HTTP {status} body: {safe_body[:500]}", file=sys.stderr)
    if classification.kind is ProviderFailureKind.INCONCLUSIVE:
        detail = safe_body[:200] if safe_body else safe_message[:200]
        try:
            pytest.skip(
                f"[{provider_id}] inconclusive provider alarm "
                f"({classification.reason}): {detail}"
            )
        except pytest.skip.Exception as skipped:
            # Carry the existing decision across pytest's outcome wrapper.
            skipped.provider_failure_classification = classification
            skipped.provider_failure_exception_type = type(exc).__name__
            raise


def official_provider_integration_job():
    return (
        os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
        and os.environ.get("GITHUB_REPOSITORY", "").strip() == "razzant/ouroboros"
    )


def require_provider_canary_credential(canary: ProviderCanary):
    if str(os.environ.get(canary.credential_env, "") or "").strip():
        return
    if canary.credential_required and official_provider_integration_job():
        pytest.fail(
            f"{canary.credential_env} is required by the official integration job "
            f"for {canary.canary_id} ({canary.model})"
        )
    policy = "required core" if canary.credential_required else "optional"
    pytest.skip(
        f"{canary.credential_env} not set; {policy} provider canary "
        f"{canary.canary_id} ({canary.model}) was not run"
    )


def full_registry_canary_tools():
    """Return the one built-in-only catalog shared with static portability tests."""
    from tests.provider_contract_catalog import shipped_builtin_tool_schemas

    tools = shipped_builtin_tool_schemas()
    names = [str((tool.get("function") or {}).get("name") or "") for tool in tools]
    assert names == sorted(names) and len(names) == len(set(names)), names
    assert CANARY_TOOL_NAME in names, names
    return copy.deepcopy(tools)


def delegate_start_canary_arguments(nonce: str):
    return {
        "prompt": (
            f"Provider contract canary {nonce}. Do not perform external actions; "
            "the returned call will be validated but never executed."
        ),
        "subagent_id": CANARY_SUBAGENT_ID,
    }


def delegate_start_canary_context():
    """Synthetic, non-executing roster that makes the requested selector well-defined.

    delegate_start takes subagent_id only for a direct fresh start, as an exact
    Agent-session actor id from Available subagents; asked for it with no roster,
    the request leaves that premise unstated. This synthetic block borrows Main's
    catalog heading and relevant row vocabulary; nothing is registered or started.
    """
    roster = {"rows": [{
        "subagent_id": CANARY_SUBAGENT_ID,
        "route_class": "Agent session",
        "recommended_use": (
            "Synthetic provider-contract canary actor. Nothing is registered or "
            "started: the returned call is validated, never executed."
        ),
    }]}
    return (
        "## Available subagents\n\n"
        + json.dumps(roster, ensure_ascii=False, indent=1)
        + "\n\nNo configured session is running and nothing is being retried: this "
        "is a direct fresh start of that Agent session actor, which requires its "
        "subagent_id. "
    )


def _delegate_start_tool(tools):
    return next(
        tool
        for tool in tools
        if str((tool.get("function") or {}).get("name") or "") == CANARY_TOOL_NAME
    )


def _named_tool_choice():
    return {"type": "function", "function": {"name": CANARY_TOOL_NAME}}


_CANARY_DIAGNOSTIC_MAX_CHARS = 160
_CANARY_DIAGNOSTIC_MAX_KEYS = 32


def _bounded_diagnostic_label(value, limit=_CANARY_DIAGNOSTIC_MAX_CHARS):
    if value is None or not isinstance(value, (str, int, float, bool)):
        return None
    try:
        value = str(value).strip()
    except (TypeError, ValueError, OverflowError):
        return None
    if not value or len(value) > limit or not value.isprintable():
        return None
    return value if sanitize_tool_result_for_log(value) == value else None


def _safe_nonnegative_int(value, maximum=10**12):
    try:
        return max(0, min(int(value or 0), maximum))
    except (TypeError, ValueError, OverflowError):
        return 0


def _text_facts(value):
    encoded = value.encode("utf-8", errors="replace")
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def _canary_evidence(canary_or_model, message, usage, *, call=None, call_index=None, parse_error=None,
                     key_diff=None):
    """Return bounded structural evidence; never copy provider payloads."""
    canary_id = canary_or_model.canary_id if isinstance(canary_or_model, ProviderCanary) else ""
    model = canary_or_model.model if isinstance(canary_or_model, ProviderCanary) else canary_or_model
    message = message if isinstance(message, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    keys = []
    message_keys_omitted = max(0, len(message) - _CANARY_DIAGNOSTIC_MAX_KEYS)
    for index, key in enumerate(message):
        if index >= _CANARY_DIAGNOSTIC_MAX_KEYS:
            break
        safe_key = _bounded_diagnostic_label(key)
        if safe_key is None:
            message_keys_omitted += 1
            continue
        keys.append(safe_key)
    keys.sort()
    message_finish = message.get("finish_reason") or message.get("stop_reason")
    response_finish_present = "response_finish_reason" in usage
    response_finish = usage.get("response_finish_reason") if response_finish_present else None
    finish = message_finish if message_finish is not None else response_finish
    evidence = {
        "canary_id": _bounded_diagnostic_label(canary_id),
        "model": _bounded_diagnostic_label(model),
        "response_id": _bounded_diagnostic_label(message.get("response_id")),
        "provider": _bounded_diagnostic_label(usage.get("provider")),
        "resolved_model": _bounded_diagnostic_label(usage.get("resolved_model")),
        "response_provider": _bounded_diagnostic_label(usage.get("response_provider")),
        "finish_reason": _bounded_diagnostic_label(finish),
        "response_finish_reason": _bounded_diagnostic_label(response_finish),
        "stop_reason": _bounded_diagnostic_label(message.get("stop_reason")),
        "message_keys": keys,
        "message_keys_omitted": message_keys_omitted,
        "prompt_tokens": _safe_nonnegative_int(usage.get("prompt_tokens")),
        "completion_tokens": _safe_nonnegative_int(usage.get("completion_tokens")),
        "ledger_attempts": len(usage.get("ledger_attempt_ids")) if isinstance(usage.get("ledger_attempt_ids"), list) else 0,
    }
    content = message.get("content")
    if isinstance(content, str):
        evidence["content_bytes"], evidence["content_sha256"] = _text_facts(content)
    elif content is not None:
        evidence["content_type"] = type(content).__name__
    if call_index is not None:
        evidence["call_index"] = _safe_nonnegative_int(call_index, 10**6)
    if isinstance(call, dict):
        function = call.get("function")
        if isinstance(function, dict):
            evidence["call_name"] = _bounded_diagnostic_label(function.get("name"))
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                evidence["arguments_bytes"], evidence["arguments_sha256"] = _text_facts(raw_arguments)
            elif raw_arguments is not None:
                evidence["arguments_type"] = type(raw_arguments).__name__
        else:
            evidence["function_type"] = type(function).__name__
    if key_diff is not None:
        extra, missing = key_diff
        evidence["extra_keys"] = [label for label in map(_bounded_diagnostic_label, extra) if label]
        evidence["missing_keys"] = [label for label in map(_bounded_diagnostic_label, missing) if label]
    if parse_error is not None:
        evidence["parse_error"] = {"type": type(parse_error).__name__}
        for field in ("pos", "lineno", "colno"):
            value = getattr(parse_error, field, None)
            if isinstance(value, int) and value >= 0:
                evidence["parse_error"][field] = min(value, 10**12)
    provider_error = usage.get("provider_error")
    if isinstance(provider_error, dict):
        evidence["provider_error"] = {
            key: _bounded_diagnostic_label(provider_error.get(key))
            for key in ("kind", "code", "type") if provider_error.get(key) is not None
        }
        if provider_error.get("message"):
            raw_provider_message = str(provider_error["message"])
            safe_provider_message = sanitize_tool_result_for_log(raw_provider_message)
            evidence["provider_error_message_bytes"], evidence["provider_error_message_sha256"] = (
                _text_facts(raw_provider_message)
            )
            if safe_provider_message != raw_provider_message:
                evidence["provider_error_message"] = "***REDACTED***"
    return evidence


def _canary_failure_payload(canary_or_model, message, usage, violation, **kwargs):
    evidence = _canary_evidence(canary_or_model, message, usage, **kwargs)
    evidence["violation"] = violation
    return {"provider_contract_violation": evidence}


def _canary_failure(canary_or_model, message, usage, violation, **kwargs):
    return AssertionError(_canary_failure_payload(
        canary_or_model, message, usage, violation, **kwargs,
    ))


def assert_openai_canary_usage(usage, model):
    def failure(code):
        return _canary_failure_payload(model, None, usage, code)
    assert isinstance(usage, dict), failure("usage_not_mapping")
    assert usage.get("provider") == "openai", failure("unexpected_accounting_provider")
    assert usage.get("resolved_model") == normalize_model_identity(model), failure("unexpected_accounting_model")
    assert _safe_nonnegative_int(usage.get("prompt_tokens")) > 0, failure("missing_prompt_tokens")
    assert _safe_nonnegative_int(usage.get("completion_tokens")) > 0, failure("missing_completion_tokens")
    assert usage.get("reasoning_effort_clamped") is None, failure("reasoning_effort_clamped")

    disclosure = usage.get("request_wire")
    assert isinstance(disclosure, dict), failure("missing_request_wire_disclosure")
    for key, expected in (("requested_effort", "medium"), ("applied_effort", "medium"),
                          ("requested_tool_dialect", "function"), ("applied_tool_dialect", "openai_chat_custom"),
                          ("reason_code", "requested_wire_form"), ("ladder_ordinal", 1)):
        assert disclosure.get(key) == expected, failure(f"request_wire_{key}")
    assert disclosure.get("task_local") is False, failure("request_wire_task_local")
    actions = disclosure.get("applied_actions")
    assert isinstance(actions, list), failure("request_wire_actions_not_list")
    for applied in actions:
        assert isinstance(applied, dict) and applied.get("source") != "task_local", failure(
            "request_wire_task_local_action",
        )
        action = applied.get("action")
        assert isinstance(action, dict), failure("request_wire_action_not_mapping")
        assert action.get("kind") != "replace_dialect", failure("request_wire_dialect_replaced")
        assert not (
            action.get("kind") == "set_value" and action.get("field") == "effort"
        ), failure("request_wire_effort_changed")
        assert not set(action.get("fields") or ()).intersection({
            "reasoning_effort", "thinking", "output_config", "extra_body.reasoning",
        }), failure("request_wire_reasoning_changed")
    assert disclosure.get("attempt_id"), failure("request_wire_attempt_missing")
    for key in (
        "source_profile_fingerprint",
        "accepted_profile_fingerprint",
        "candidate_sha256",
    ):
        value = str(disclosure.get(key) or "")
        assert len(value) == 64 and not set(value.lower()) - set("0123456789abcdef"), failure(
            f"request_wire_{key}",
        )
    return disclosure


def assert_canary_usage(usage, canary: ProviderCanary, *, forced_tool_choice: bool = False):
    def failure(code):
        return _canary_failure_payload(canary, None, usage, code)
    assert isinstance(usage, dict), failure("usage_not_mapping")
    assert usage.get("provider") == canary.expected_provider, failure("unexpected_accounting_provider")
    assert usage.get("resolved_model") == normalize_model_identity(canary.model), failure("unexpected_accounting_model")
    if (isinstance(usage.get("stream_receipt"), dict) and not _safe_nonnegative_int(usage.get("prompt_tokens"))
            and not _safe_nonnegative_int(usage.get("completion_tokens"))):
        # A complete streamed reply whose route sent no final usage frame (a provider
        # that drops stream_options under wire recovery, or ignores include_usage):
        # money is unknown, the contract is not broken — weather, recorded as a
        # warning through the existing telemetry, never RED.
        warnings = usage.setdefault("canary_warnings", [])
        if isinstance(warnings, list) and len(warnings) < 4:
            warnings.append({**_canary_evidence(canary, None, usage), "code": "streamed_reply_without_usage_frame"})
    else:
        assert _safe_nonnegative_int(usage.get("prompt_tokens")) > 0, failure("missing_prompt_tokens")
        assert _safe_nonnegative_int(usage.get("completion_tokens")) > 0, failure("missing_completion_tokens")
    if canary.reasoning_effort == "medium":
        expected_effort = canary.reasoning_effort
        if canary.expected_provider == "deepseek":
            # DeepSeek's wire enum aliases medium to high, and its thinking
            # mode rejects a forced tool choice, so the named first turn runs
            # with thinking disabled; both projections must be disclosed and
            # the physical payload carries the projected tier.
            if forced_tool_choice:
                expected_effort, reason = "none", "provider_forced_tool_choice"
            else:
                expected_effort = normalize_deepseek_reasoning_effort(expected_effort)
                reason = "provider_wire_mapping"
            note = usage.get("reasoning_effort_clamped")
            assert isinstance(note, dict), failure("reasoning_effort_clamped")
            assert note.get("requested") == canary.reasoning_effort, failure("reasoning_effort_clamped_requested")
            assert note.get("applied") == expected_effort, failure("reasoning_effort_clamped_applied")
            assert note.get("reason") == reason, failure("reasoning_effort_clamped_reason")
        else:
            assert usage.get("reasoning_effort_clamped") is None, failure("reasoning_effort_clamped")
        disclosure = usage.get("request_wire")
        assert isinstance(disclosure, dict), failure("missing_request_wire_disclosure")
        assert disclosure.get("requested_effort") == expected_effort, failure("request_wire_requested_effort")
        assert disclosure.get("applied_effort") == expected_effort, failure("request_wire_applied_effort")
    if canary.expected_provider == "openai":
        return assert_openai_canary_usage(usage, canary.model)
    return usage.get("request_wire")


def _semantic_empty_canary_message(message) -> bool:
    return (
        isinstance(message, dict)
        and not (message.get("tool_calls") or [])
        and not str(message.get("content") or "").strip()
    )


def _safe_empty_canary_diagnostic(canary: ProviderCanary, message, usage, attempts: int):
    diagnostic = _canary_evidence(canary, message, usage)
    diagnostic["attempts"] = _safe_nonnegative_int(attempts, 10**6)
    return diagnostic


def _record_canary_response_warnings(canary: ProviderCanary, message, usage):
    """Record tolerated response-shape drift without copying provider text."""
    if not isinstance(usage, dict):
        return
    # The usage mapping can contain provider-controlled extension fields. Never
    # preserve a provider-supplied warning list for the host-owned CI emitter.
    warnings = []
    usage["canary_warnings"] = warnings
    if not isinstance(message, dict):
        return
    calls = message.get("tool_calls")
    content = message.get("content")
    if not isinstance(calls, list) or not calls or not isinstance(content, str) or not content.strip():
        return
    warning = _canary_evidence(canary, message, usage)
    warning["code"] = "native_tool_call_with_assistant_text"
    if len(warnings) < 4:
        warnings.append(warning)


def _emit_canary_response_warnings(usage):
    """Expose bounded mixed-content warnings on the trusted CI test surface."""
    import warnings

    if not isinstance(usage, dict):
        return
    entries = usage.get("canary_warnings")
    if not isinstance(entries, list):
        return
    for warning in entries[:4]:
        if not isinstance(warning, dict):
            continue
        warnings.warn(
            "provider_canary_warning " + json.dumps(
                warning, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def _observe_canary(observer, event, **facts):
    """A diagnostic observer cannot change the canary's calls or verdict."""
    if observer is not None:
        try:
            observer.observe(event, **facts)
        except Exception as exc:
            # Never expose the exception's provider-controlled text.
            print(f"provider diagnostics_incomplete: {type(exc).__name__}")


def _chat_canary_turn(client, *, canary: ProviderCanary, chat_kwargs, observer=None, turn="tool"):
    """Retry one runtime-classified semantic-empty turn on the exact same route."""
    message = None
    usage = {}
    attempts = 0
    for attempt in range(CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS):
        attempts = attempt + 1
        attempt_kwargs = copy.deepcopy(chat_kwargs)
        attempt_kwargs["bypass_response_cache"] = attempt > 0
        try:
            message, usage = client.chat(**attempt_kwargs)
        except BaseException as exc:
            _observe_canary(observer, "attempt", turn=turn, ordinal=attempts,
                            request=attempt_kwargs, error=exc)
            raise
        _observe_canary(observer, "attempt", turn=turn, ordinal=attempts,
                        request=attempt_kwargs, message=message, usage=usage,
                        semantic_empty=_semantic_empty_canary_message(message))
        if not _semantic_empty_canary_message(message):
            return message, usage

        from ouroboros.loop_llm_call import _classify_empty_response

        event_type, _is_provider_glitch, permanent_body_error = _classify_empty_response(
            usage, message,
        )
        if permanent_body_error or event_type == "remote_context_overflow":
            break
        if attempts < CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS:
            time.sleep(CANARY_EMPTY_RESPONSE_BACKOFF_SEC)

    raise AssertionError({
        "semantic_empty_provider_response": _safe_empty_canary_diagnostic(
            canary, message, usage, attempts,
        ),
    })


def assert_normalized_canary_call(
    message,
    tools,
    requested_arguments,
    *,
    canary=None,
    usage=None,
):
    """Every native call must be schema-valid and carry EXACTLY the requested object.

    An extra declared key is a RED contract fact, not tolerated noise: a route that
    forces optional properties (issue #1411) fills keys the prompt never asked for.
    """
    from jsonschema import validators

    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or not calls:
        raise _canary_failure(canary or "", message, usage, "missing_tool_calls")
    schema = _delegate_start_tool(tools)["function"]["parameters"]
    validator_class = validators.validator_for(schema)
    validator_class.check_schema(schema)
    seen_ids = set()
    for call_index, call in enumerate(calls):
        if not isinstance(call, dict):
            raise _canary_failure(
                canary or "", message, usage, "tool_call_not_mapping",
                call_index=call_index,
            )
        if call.get("type") != "function":
            raise _canary_failure(
                canary or "", message, usage, "tool_call_type",
                call=call, call_index=call_index,
            )
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id != call_id.strip():
            raise _canary_failure(
                canary or "", message, usage, "tool_call_id",
                call=call, call_index=call_index,
            )
        if call_id in seen_ids:
            raise _canary_failure(
                canary or "", message, usage, "duplicate_tool_call_id",
                call=call, call_index=call_index,
            )
        seen_ids.add(call_id)
        function = call.get("function")
        if not isinstance(function, dict):
            raise _canary_failure(
                canary or "", message, usage, "function_not_mapping",
                call=call, call_index=call_index,
            )
        if function.get("name") != CANARY_TOOL_NAME:
            raise _canary_failure(
                canary or "", message, usage, "unexpected_tool_name",
                call=call, call_index=call_index,
            )
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise _canary_failure(
                canary or "", message, usage, "arguments_not_string",
                call=call, call_index=call_index,
            )
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError) as error:
            raise _canary_failure(
                canary or "", message, usage, "malformed_arguments_json",
                call=call, call_index=call_index, parse_error=error,
            ) from error
        schema_errors = list(validator_class(schema).iter_errors(arguments))
        if schema_errors:
            raise _canary_failure(
                canary or "", message, usage, "arguments_schema",
                call=call, call_index=call_index,
            )
        if not isinstance(arguments, dict):
            raise _canary_failure(
                canary or "", message, usage, "arguments_not_object",
                call=call, call_index=call_index,
            )
        unknown_keys = set(arguments) - set((schema.get("properties") or {}).keys())
        if unknown_keys:
            raise _canary_failure(
                canary or "", message, usage, "arguments_unknown_keys",
                call=call, call_index=call_index,
            )
        if set(arguments) != set(requested_arguments):
            # Only declared property names reach here, so naming them is safe evidence.
            raise _canary_failure(
                canary or "", message, usage, "arguments_exact_keys",
                call=call, call_index=call_index, key_diff=(
                    sorted(set(arguments) - set(requested_arguments)),
                    sorted(set(requested_arguments) - set(arguments)),
                ),
            )
        for key, expected in requested_arguments.items():
            if arguments.get(key) != expected:
                raise _canary_failure(
                    canary or "", message, usage, f"arguments_{key}",
                    call=call, call_index=call_index,
                )
    return calls


def run_provider_contract_canary(
    client,
    *,
    canary: ProviderCanary,
    tools,
    nonce: str,
    observer=None,
):
    """Exercise the public chat seam without executing the returned tool call."""
    requested_arguments = delegate_start_canary_arguments(nonce)
    arguments_json = json.dumps(requested_arguments, ensure_ascii=False, sort_keys=True)
    final_marker = f"FULL_REGISTRY_CONTINUED_{nonce}"
    _observe_canary(observer, "expected", arguments=requested_arguments,
                    final_marker=final_marker if canary.continue_to_final else None)
    continuation_instruction = (
        "After its tool result, read the expected_final_marker field and reply "
        "with exactly that value. "
        if canary.continue_to_final
        else ""
    )
    conversation = [{
        "role": "user",
        "content": (
            f"{delegate_start_canary_context()}"
            f"Call {CANARY_TOOL_NAME} exactly once with exactly this JSON object "
            f"as its arguments: {arguments_json}. Do not add, omit, or change a field. "
            f"{continuation_instruction}"
            "Return the tool call now and no prose."
        ),
    }]
    tool_choice = _named_tool_choice() if canary.named_tool_choice else "auto"
    from ouroboros.provider_models import tool_schema_limit
    from ouroboros.tool_policy import fit_tool_schemas_to_limit

    # The route's physical ceiling applies as in Main (loop_model_call._fit_route_tool_ceiling);
    # the requested tool stays loaded like a schema the actor loaded on purpose.
    sent_tools, _left_out = fit_tool_schemas_to_limit(
        tools, tool_schema_limit(canary.model), keep=(CANARY_TOOL_NAME,),
    )
    message, usage = _chat_canary_turn(
        client,
        canary=canary,
        observer=observer,
        chat_kwargs={
            "messages": conversation,
            "model": canary.model,
            "tools": copy.deepcopy(sent_tools),
            "tool_choice": tool_choice,
            "reasoning_effort": canary.reasoning_effort,
            "max_tokens": CANARY_MAX_TOKENS,
            # Main sets stream=True on every remote completion, so the canary's
            # first turn must exercise that transport rather than one Main never
            # uses. Rows that continue to a final answer keep their second turn
            # non-stream, so those routes (and GigaChat, whose lane drops the
            # flag) still cover the non-stream transport at zero extra physical
            # sends; every other row now covers streaming only.
            "stream": True,
            "no_proxy": True,
            "timeout": CANARY_TIMEOUT_SEC,
        },
    )
    # The provider must return schema-valid native calls whose arguments are
    # exactly the requested object: the nonce-bearing prompt and the subagent_id
    # selector, no omitted, substituted or provider-filled optional key. Valid
    # assistant text beside those calls is tolerated as bounded warning telemetry.
    calls = assert_normalized_canary_call(
        message,
        tools,
        requested_arguments,
        canary=canary,
        usage=usage,
    )
    _record_canary_response_warnings(canary, message, usage)
    first_disclosure = assert_canary_usage(
        usage, canary, forced_tool_choice=canary.named_tool_choice,
    )
    if not canary.continue_to_final:
        return message, usage, None, None

    continuation = [
        *conversation,
        message,
        *[
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps({
                    "ok": True,
                    "nonce": nonce,
                    "expected_final_marker": final_marker,
                }, sort_keys=True),
            }
            for call in calls
        ],
    ]
    final_message, final_usage = _chat_canary_turn(
        client,
        canary=canary,
        observer=observer,
        turn="continuation",
        chat_kwargs={
            "messages": continuation,
            "model": canary.model,
            "tools": [copy.deepcopy(_delegate_start_tool(tools))],
            "tool_choice": "none",
            "reasoning_effort": canary.reasoning_effort,
            "max_tokens": CANARY_CONTINUATION_MAX_TOKENS,
            "no_proxy": True,
            "timeout": CANARY_TIMEOUT_SEC,
        },
    )
    final_disclosure = assert_canary_usage(final_usage, canary)
    if isinstance(first_disclosure, dict) and isinstance(final_disclosure, dict):
        if final_disclosure.get("candidate_sha256") == first_disclosure.get(
            "candidate_sha256"
        ):
            raise _canary_failure(
                canary,
                final_message,
                final_usage,
                "continuation_candidate_binding",
            )
    if not isinstance(final_message, dict) or final_message.get("tool_calls"):
        raise _canary_failure(
            canary,
            final_message,
            final_usage,
            "unexpected_continuation_tool_calls",
        )
    if str(final_message.get("content") or "").strip() != final_marker:
        raise _canary_failure(
            canary,
            final_message,
            final_usage,
            "continuation_marker",
        )
    return message, usage, final_message, final_usage
