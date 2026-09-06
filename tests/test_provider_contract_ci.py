"""Secretless CI contracts for the trusted full-registry provider alarms."""

from __future__ import annotations

import contextlib
import copy
import io
import json

import pytest

from ouroboros.provider_models import (
    OPENAI_DIRECT_DEFAULTS,
    normalize_deepseek_reasoning_effort,
    normalize_model_identity,
)
from ouroboros.request_wire_contract import canonical_sha256
from ouroboros.request_wire_receipts import (
    WireCandidateSpec,
    bind_wire_candidate,
    observe_wire_semantics,
)
from ouroboros.usage_accounting import PhysicalAttemptCapture
from tests.provider_contract_ci import (
    CANARY_CONTINUATION_MAX_TOKENS,
    CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS,
    CANARY_MAX_TOKENS,
    CANARY_TIMEOUT_SEC,
    CANARY_TOOL_NAME,
    ProviderCanary,
    ProviderFailureClassification,
    ProviderFailureKind,
    _emit_canary_response_warnings,
    assert_normalized_canary_call,
    assert_openai_canary_usage,
    classify_provider_failure,
    delegate_start_canary_arguments,
    full_registry_canary_tools,
    provider_canary_matrix,
    require_provider_canary_credential,
    run_provider_contract_canary,
    skip_on_provider_environmental_error,
    unique_openai_direct_defaults,
)


def _http_error(status_code: int, body: str):
    class Response:
        pass

    response = Response()
    response.status_code = status_code
    response.text = body
    exc = RuntimeError(f"provider returned HTTP {status_code}")
    exc.response = response
    return exc


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (
            400,
            (
                '{"error":{"message":"Function tools with reasoning_effort are not '
                'supported. Use responses or set reasoning_effort to none."}}'
            ),
        ),
        (400, '{"error":{"message":"tools.19.custom.input_schema is invalid"}}'),
        (400, '{"error":{"message":"Unknown parameter: tools[0].custom"}}'),
        (401, '{"error":{"code":"invalid_api_key"}}'),
        (403, '{"message":"API key verification failed: key is expired"}'),
        (404, '{"error":{"code":"model_not_found"}}'),
        (422, '{"error":{"message":"reasoning_effort medium is unsupported"}}'),
    ],
)
def test_contract_and_auth_4xx_are_red(status_code, body):
    exc = _http_error(status_code, body)
    assert classify_provider_failure(
        "provider_canary", exc,
    ) == ProviderFailureClassification(
        ProviderFailureKind.RED,
        "provider_contract_or_unclassified",
        status_code,
    )
    assert skip_on_provider_environmental_error("provider_canary", exc) is None


@pytest.mark.parametrize(
    ("status_code", "body", "reason"),
    [
        (429, '{"error":{"code":"rate_limit_exceeded"}}', "rate_limit_429"),
        (503, '{"error":{"message":"upstream unavailable"}}', "provider_5xx"),
        (400, '{"error":{"code":"insufficient_quota"}}', "quota_or_billing"),
        (402, '{"error":{"message":"credit balance is too low"}}', "quota_or_billing"),
        (
            402,
            '{"error":{"message":"This request requires more credits, or fewer '
            'max_tokens. You can only afford 348."}}',
            "quota_or_billing",
        ),
    ],
)
def test_only_explicit_environmental_outcomes_are_inconclusive(
    status_code,
    body,
    reason,
):
    classification = classify_provider_failure(
        "provider_canary", _http_error(status_code, body),
    )
    assert classification.kind is ProviderFailureKind.INCONCLUSIVE
    assert classification.reason == reason
    assert classification.status_code == status_code
    with pytest.raises(pytest.skip.Exception, match=reason):
        skip_on_provider_environmental_error(
            "provider_canary", _http_error(status_code, body),
        )


def test_timeout_is_inconclusive_but_http_400_with_timeout_cause_is_red():
    timeout = TimeoutError("timed out")
    assert classify_provider_failure(
        "provider_canary", timeout,
    ) == ProviderFailureClassification(
        ProviderFailureKind.INCONCLUSIVE,
        "transport_timeout",
    )
    with pytest.raises(pytest.skip.Exception, match="transport_timeout"):
        skip_on_provider_environmental_error("provider_canary", timeout)

    http_400 = _http_error(400, '{"error":{"message":"reasoning timeout invalid"}}')
    http_400.__cause__ = TimeoutError("socket timed out")
    assert classify_provider_failure(
        "provider_canary", http_400,
    ).kind is ProviderFailureKind.RED


def test_disconnect_and_generic_connection_error_are_red():
    disconnect = RuntimeError("APIConnectionError: Connection error.")
    disconnect.__cause__ = RuntimeError(
        "httpx.RemoteProtocolError: Server disconnected without sending a response."
    )
    assert classify_provider_failure(
        "provider_canary", disconnect,
    ).kind is ProviderFailureKind.RED
    assert classify_provider_failure(
        "provider_canary", RuntimeError("APIConnectionError: Connection error."),
    ).kind is ProviderFailureKind.RED


def test_provider_alarm_output_sanitizes_token_shaped_evidence(capsys):
    sentinel = "sk-proj-" + ("A" * 40)
    exc = _http_error(429, f'{{"error":{{"token":"{sentinel}"}}}}')
    with pytest.raises(pytest.skip.Exception) as caught:
        skip_on_provider_environmental_error("provider_canary", exc)

    assert sentinel not in capsys.readouterr().err
    assert sentinel not in str(caught.value)
    assert "***REDACTED***" in str(caught.value)

    with pytest.raises(pytest.skip.Exception) as caught_message:
        skip_on_provider_environmental_error(
            "provider_canary",
            TimeoutError(f"timed out with token {sentinel}"),
        )
    assert sentinel not in str(caught_message.value)
    assert "***REDACTED***" in str(caught_message.value)


def test_exact_provider_canary_matrix_logical_turns_and_attempt_bound():
    matrix = provider_canary_matrix()
    assert [(row.canary_id, row.model) for row in matrix] == [
        ("openrouter_gemini", "google/gemini-3.8-flash"),
        ("openrouter_opus", "anthropic/claude-opus-5"),
        ("openrouter_fable", "anthropic/claude-fable-5.1"),
        ("openrouter_gpt", "openai/gpt-5.6-luna"),
        ("openrouter_grok", "x-ai/grok-4.6"),
        ("openrouter_deepseek", "deepseek/deepseek-v4-pro-0813"),
        ("openai_direct_main", "openai::gpt-5.6-terra"),
        ("openai_direct_light", "openai::gpt-5.6-luna"),
        ("openai_direct_fallback", "openai::gpt-5.6-sol"),
        ("anthropic_direct", "anthropic::claude-sonnet-5"),
        ("minimax_direct", "minimax::MiniMax-M3"),
        ("deepseek_direct", "deepseek::deepseek-v4-flash"),
        ("cloudru_direct", "cloudru::zai-org/GLM-4.7"),
        ("gigachat_direct", "gigachat::GigaChat-2-Max"),
    ]
    medium_ids = {row.canary_id for row in matrix if row.reasoning_effort == "medium"}
    assert medium_ids == {
        "openrouter_gemini",
        "openrouter_opus",
        "openrouter_fable",
        "openrouter_gpt",
        "openrouter_grok",
        "openrouter_deepseek",
        "openai_direct_main",
        "openai_direct_light",
        "openai_direct_fallback",
        "anthropic_direct",
        "deepseek_direct",
    }
    assert [row.canary_id for row in matrix if row.continue_to_final] == [
        "openai_direct_main", "deepseek_direct"
    ]
    assert [row.canary_id for row in matrix if not row.named_tool_choice] == [
        "openrouter_fable", "gigachat_direct"
    ]
    logical_turns = sum(1 + int(row.continue_to_final) for row in matrix)
    assert logical_turns == 16
    assert logical_turns * CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS == 32
    assert sum(
        1 + int(row.continue_to_final)
        for row in matrix
        if row.credential_required
    ) == 11


def test_direct_anthropic_named_tool_choice_projects_without_type_error():
    from ouroboros.llm import LLMClient

    named = {"type": "function", "function": {"name": CANARY_TOOL_NAME}}
    assert LLMClient._build_anthropic_tool_choice(named) == {
        "type": "tool", "name": CANARY_TOOL_NAME,
    }
    assert LLMClient._build_anthropic_tool_choice("required") == {"type": "any"}
    assert LLMClient._build_anthropic_tool_choice("none") == {"type": "none"}


def test_openai_models_follow_default_ssot_without_allowlist(monkeypatch):
    expected = tuple(dict.fromkeys(model for model in OPENAI_DIRECT_DEFAULTS.values() if model))
    assert unique_openai_direct_defaults() == expected
    assert OPENAI_DIRECT_DEFAULTS["main"] in expected
    assert len(expected) == len(set(expected))
    assert all(model.startswith("openai::") for model in expected)

    future = "openai::future-direct-model"
    monkeypatch.setitem(OPENAI_DIRECT_DEFAULTS, "future_ci_role", future)
    monkeypatch.setitem(OPENAI_DIRECT_DEFAULTS, "future_ci_duplicate", future)
    assert unique_openai_direct_defaults()[-1] == future
    assert unique_openai_direct_defaults().count(future) == 1
    assert [
        row.model
        for row in provider_canary_matrix()
        if row.expected_provider == "openai"
    ][-1] == future


def test_required_and_optional_credential_policy(monkeypatch):
    required = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_gemini")
    optional = next(row for row in provider_canary_matrix() if row.canary_id == "minimax_direct")
    monkeypatch.delenv(required.credential_env, raising=False)
    monkeypatch.delenv(optional.credential_env, raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REPOSITORY", "razzant/ouroboros")

    with pytest.raises(pytest.fail.Exception, match="required"):
        require_provider_canary_credential(required)
    with pytest.raises(pytest.skip.Exception, match="optional provider canary"):
        require_provider_canary_credential(optional)

    monkeypatch.setenv("GITHUB_REPOSITORY", "fork/ouroboros")
    with pytest.raises(pytest.skip.Exception, match="required core"):
        require_provider_canary_credential(required)

    monkeypatch.setenv(required.credential_env, "test-present")
    assert require_provider_canary_credential(required) is None


def _bind_candidate(model, tools, messages, tool_choice):
    prefix, separator, resolved_model = model.partition("::")
    assert (prefix, separator) == ("openai", "::")
    source = {
        "model": resolved_model,
        "messages": copy.deepcopy(messages),
        "reasoning_effort": "medium",
        "max_completion_tokens": CANARY_MAX_TOKENS,
        "tools": copy.deepcopy(tools),
        "tool_choice": copy.deepcopy(tool_choice),
    }
    return bind_wire_candidate(
        target={
            "provider": "openai",
            "resolved_model": resolved_model,
            "usage_model": normalize_model_identity(model),
            "base_url": "https://api.openai.com/v1",
        },
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "medium", "requested_wire_form",
        ),
        requested_effort="medium",
        ladder_ordinal=1,
    )


def test_shipped_defaults_bind_full_registry_custom_medium_shape():
    tools = full_registry_canary_tools()
    names = [tool["function"]["name"] for tool in tools]
    choice = {"type": "function", "function": {"name": CANARY_TOOL_NAME}}
    delegate_schema = next(
        tool["function"]["parameters"]
        for tool in tools
        if tool["function"]["name"] == CANARY_TOOL_NAME
    )
    for model in unique_openai_direct_defaults():
        candidate = _bind_candidate(
            model,
            tools,
            [{"role": "user", "content": "Call delegate_start exactly once."}],
            choice,
        )
        physical = candidate.physical_payload()
        assert candidate.source_profile.tool_dialect == "function"
        assert candidate.accepted_profile.tool_dialect == "openai_chat_custom"
        assert candidate.accepted_profile.reasoning_carrier == "reasoning_effort"
        assert candidate.candidate_spec.reason_code == "requested_wire_form"
        assert candidate.physical_model == normalize_model_identity(model)
        assert physical["model"] == model.split("::", 1)[-1]
        assert physical["reasoning_effort"] == "medium"
        assert physical["max_completion_tokens"] == CANARY_MAX_TOKENS
        assert "max_tokens" not in physical
        assert physical["tool_choice"] == {
            "type": "custom", "custom": {"name": CANARY_TOOL_NAME},
        }
        assert [tool["custom"]["name"] for tool in physical["tools"]] == names
        assert candidate.custom_catalog.tool_names == tuple(names)
        assert candidate.custom_catalog.schema_binding(
            CANARY_TOOL_NAME
        ).schema() == delegate_schema


def _canonical_canary_call(call_id, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": CANARY_TOOL_NAME,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_remote_normalizer_keeps_outer_finish_reason_as_usage_fact(finish_reason):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    message, usage = client._normalize_remote_response(
        {
            "id": "response-finish-fact",
            "provider": "openrouter/upstream-a",
            "choices": [{
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": "ok"},
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        {
            "provider": "openrouter",
            "usage_model": "x-ai/grok-4.6",
            "supports_openrouter_extensions": False,
        },
        skip_cost_fetch=True,
    )

    assert usage["response_finish_reason"] == finish_reason
    assert usage["response_provider"] == "openrouter/upstream-a"
    assert message["response_id"] == "response-finish-fact"
    assert "finish_reason" not in message


def test_malformed_native_arguments_fail_closed_with_bounded_evidence():
    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    nonce = "malformed-native"
    secret = "sk-or-v1-" + "A" * 40

    class MalformedNativeClient:
        def __init__(self):
            self.sends = 0

        def chat(self, **kwargs):
            self.sends += 1
            raw = '{"prompt":"' + secret
            return {
                "content": "provider explanation",
                "tool_calls": [{
                    "id": "bad-native",
                    "type": "function",
                    "function": {"name": CANARY_TOOL_NAME, "arguments": raw},
                }],
            }, {**_fake_usage(canary, self.sends), "provider_error": {"code": "insufficient_quota"}}

    client = MalformedNativeClient()
    with pytest.raises(AssertionError) as caught:
        run_provider_contract_canary(
            client, canary=canary, tools=full_registry_canary_tools(), nonce=nonce,
        )

    evidence = caught.value.args[0]["provider_contract_violation"]
    assert evidence["violation"] == "malformed_arguments_json"
    assert evidence["parse_error"]["type"] == "JSONDecodeError"
    assert evidence["arguments_bytes"] == len(("{\"prompt\":\"" + secret).encode())
    assert len(evidence["arguments_sha256"]) == 64
    assert secret not in str(caught.value)
    assert skip_on_provider_environmental_error(canary.canary_id, caught.value) is None
    assert client.sends == 1


def test_native_call_with_text_is_tolerated_and_warns_without_copying_text():
    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    nonce = "mixed-native-text"
    expected = delegate_start_canary_arguments(nonce)
    prose = "provider explanation " + ("P" * 5000)

    class MixedClient:
        def chat(self, **kwargs):
            return {
                "content": prose,
                "tool_calls": [_canonical_canary_call("mixed-native", expected)],
            }, _fake_usage(canary, 1)

    _message, usage, _final, _final_usage = run_provider_contract_canary(
        MixedClient(), canary=canary, tools=full_registry_canary_tools(), nonce=nonce,
    )
    warning = usage["canary_warnings"][0]
    assert warning["code"] == "native_tool_call_with_assistant_text"
    assert warning["content_bytes"] == len(prose.encode())
    assert warning["content_sha256"]
    assert prose not in str(warning)


def test_empty_canary_diagnostic_uses_usage_finish_reason():
    import tests.provider_contract_ci as contract

    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    diagnostic = contract._safe_empty_canary_diagnostic(
        canary,
        {"content": "", "tool_calls": []},
        {**_fake_usage(canary, 1), "response_finish_reason": "length"},
        1,
    )
    assert diagnostic["finish_reason"] == "length"
    assert diagnostic["response_finish_reason"] == "length"
    distinct = contract._safe_empty_canary_diagnostic(
        canary,
        {"content": "", "tool_calls": [], "finish_reason": "message-stop"},
        {**_fake_usage(canary, 1), "response_finish_reason": "outer-length"},
        1,
    )
    assert (distinct["finish_reason"], distinct["response_finish_reason"]) == ("message-stop", "outer-length")


def test_canary_diagnostics_redact_labels_and_omit_provider_prose():
    import tests.provider_contract_ci as contract

    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    secret = "sk-or-v1-" + "B" * 40
    diagnostic = contract._safe_empty_canary_diagnostic(
        canary,
        {
            "content": "",
            "tool_calls": [],
            secret: "provider-controlled key name",
            "safe_extra": "provider-controlled but safe key",
        },
        {
            **_fake_usage(canary, 1),
            "response_provider": secret,
            "provider_error": {
                "kind": "provider_error",
                "code": "400",
                "message": "ordinary provider detail " + ("Q" * 500),
            },
        },
        1,
    )
    assert diagnostic["response_provider"] is None
    assert "provider_error_message" not in diagnostic
    assert diagnostic["provider_error_message_bytes"] > 200
    assert len(diagnostic["provider_error_message_sha256"]) == 64
    assert secret not in str(diagnostic)
    assert secret not in diagnostic["message_keys"]
    assert "safe_extra" in diagnostic["message_keys"]
    assert diagnostic["message_keys_omitted"] == 1


def test_canary_warning_emission_is_bounded_and_observable():
    warning = {
        "code": "native_tool_call_with_assistant_text",
        "content_bytes": 5000,
        "content_sha256": "a" * 64,
    }
    with pytest.warns(RuntimeWarning, match="provider_canary_warning") as caught:
        _emit_canary_response_warnings({"canary_warnings": [warning]})
    assert len(caught) == 1 and "5000" in str(caught[0].message)
    assert "provider explanation" not in str(caught[0].message)


def test_provider_warning_extension_is_not_emitted_as_host_telemetry():
    import tests.provider_contract_ci as contract

    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    secret = "provider-secret-" + ("X" * 600)
    usage = {"canary_warnings": [{"raw": secret}]}
    contract._record_canary_response_warnings(
        canary,
        {"content": "", "tool_calls": []},
        usage,
    )
    output = io.StringIO()
    with contextlib.redirect_stderr(output):
        _emit_canary_response_warnings(usage)
    assert secret not in output.getvalue()
    assert usage["canary_warnings"] == []


def test_malformed_openai_usage_assertions_keep_bounded_violation_evidence():
    canary = next(row for row in provider_canary_matrix() if row.expected_provider == "openai")
    usage = _fake_usage(canary, 1)
    usage["request_wire"]["applied_actions"] = [{
        "source": "task_local",
        "action": {"kind": "drop_field", "fields": []},
    }]
    with pytest.raises(AssertionError) as caught:
        assert_openai_canary_usage(usage, canary.model)
    evidence = caught.value.args[0]["provider_contract_violation"]
    assert evidence["violation"] == "request_wire_task_local_action"
    usage["request_wire"]["task_local"] = 0
    with pytest.raises(AssertionError) as caught:
        assert_openai_canary_usage(usage, canary.model)
    evidence = caught.value.args[0]["provider_contract_violation"]
    assert evidence["violation"] == "request_wire_task_local"


def test_custom_call_normalizes_and_replays_role_tool_continuation():
    from ouroboros.openai_chat_custom import normalize_openai_custom_tool_calls

    model = unique_openai_direct_defaults()[0]
    tools = full_registry_canary_tools()
    expected = delegate_start_canary_arguments("deterministic-custom")
    user = {"role": "user", "content": "Call delegate_start exactly once."}
    first = _bind_candidate(
        model,
        tools,
        [user],
        {"type": "function", "function": {"name": CANARY_TOOL_NAME}},
    )
    raw_arguments = json.dumps(expected, sort_keys=True)
    canonical_calls, receipts = normalize_openai_custom_tool_calls(
        [{
            "id": "call_full_registry",
            "type": "custom",
            "custom": {"name": CANARY_TOOL_NAME, "input": raw_arguments},
        }],
        first,
    )
    assert receipts[0].allows_execution
    assert canonical_calls[0] == _canonical_canary_call(
        "call_full_registry", expected,
    )

    continuation = _bind_candidate(
        model,
        tools,
        [
            user,
            {"role": "assistant", "content": None, "tool_calls": canonical_calls},
            {
                "role": "tool",
                "tool_call_id": "call_full_registry",
                "content": json.dumps({"nonce": "deterministic-custom"}),
            },
        ],
        "none",
    )
    physical = continuation.physical_payload()
    assert physical["messages"][1]["tool_calls"][0]["type"] == "custom"
    assert physical["messages"][1]["tool_calls"][0]["custom"]["input"] == raw_arguments
    assert physical["messages"][2]["role"] == "tool"
    assert physical["messages"][2]["tool_call_id"] == "call_full_registry"
    assert physical["tool_choice"] == "none"
    assert physical["reasoning_effort"] == "medium"


def _fake_usage(canary: ProviderCanary, ordinal: int):
    usage = {
        "provider": canary.expected_provider,
        "resolved_model": normalize_model_identity(canary.model),
        "prompt_tokens": 120,
        "completion_tokens": 12,
    }
    if canary.expected_provider != "openai":
        if canary.reasoning_effort == "medium":
            applied_effort = canary.reasoning_effort
            if canary.expected_provider == "deepseek":
                forced = ordinal == 1 and canary.named_tool_choice
                applied_effort = (
                    "none" if forced else normalize_deepseek_reasoning_effort(applied_effort)
                )
                usage["reasoning_effort_clamped"] = {
                    "requested": canary.reasoning_effort,
                    "applied": applied_effort,
                    "reason": "provider_forced_tool_choice" if forced else "provider_wire_mapping",
                    "model": canary.model.split("::", 1)[-1],
                }
            usage["request_wire"] = {
                "requested_effort": applied_effort,
                "applied_effort": applied_effort,
                # A continuation must bind a fresh physical candidate.
                "candidate_sha256": ("c" if ordinal == 1 else "d") * 64,
            }
        return usage
    usage["request_wire"] = {
        "requested_effort": "medium",
        "applied_effort": "medium",
        "requested_tool_dialect": "function",
        "applied_tool_dialect": "openai_chat_custom",
        "reason_code": "requested_wire_form",
        "source_profile_fingerprint": "a" * 64,
        "accepted_profile_fingerprint": "b" * 64,
        "attempt_id": f"attempt-{ordinal}",
        "candidate_sha256": ("c" if ordinal == 1 else "d") * 64,
        "ladder_ordinal": 1,
        "applied_actions": [],
        "task_local": False,
    }
    return usage


def test_reasoning_integrity_canary_rejects_explicit_none_and_clamp():
    model = unique_openai_direct_defaults()[0]
    canary = next(row for row in provider_canary_matrix() if row.model == model)
    explicit_none = _fake_usage(canary, 1)
    explicit_none["request_wire"]["applied_effort"] = "none"
    with pytest.raises(AssertionError):
        assert_openai_canary_usage(explicit_none, model)

    clamped = _fake_usage(canary, 1)
    clamped["reasoning_effort_clamped"] = {
        "requested": "medium",
        "applied": "none",
        "reason": "task_local_availability_fallback",
    }
    with pytest.raises(AssertionError):
        assert_openai_canary_usage(clamped, model)

    neutral_repair = _fake_usage(canary, 1)
    neutral_repair["request_wire"]["applied_actions"] = [{
        "source": "pending",
        "profile_fingerprint": "e" * 64,
        "action": {
            "kind": "drop_field",
            "fields": ["temperature"],
            "reason_code": "provider_unsupported_field",
        },
    }]
    assert_openai_canary_usage(neutral_repair, model)


def test_normalized_calls_require_unique_schema_valid_delegate_start():
    tools = full_registry_canary_tools()
    expected = delegate_start_canary_arguments("normalized-call")
    required = {"prompt": expected["prompt"]}
    assert assert_normalized_canary_call(
        {"tool_calls": [_canonical_canary_call("call-ok", expected)]},
        tools,
        required,
    )[0]["id"] == "call-ok"

    provider_defaults = {
        **expected,
        "root": "skill_payload",
        "bucket": "",
        "skill_name": "",
        "retry_of": "",
        "max_seconds": 0,
    }
    assert assert_normalized_canary_call(
        {"tool_calls": [_canonical_canary_call("call-defaults", provider_defaults)]},
        tools,
        required,
    )[0]["id"] == "call-defaults"

    provider_selector = {
        "prompt": expected["prompt"],
        "retry_of": "provider-contract-canary",
        "root": "skill_payload",
        "bucket": "external",
    }
    assert assert_normalized_canary_call(
        {"tool_calls": [_canonical_canary_call("call-selector", provider_selector)]},
        tools,
        required,
    )[0]["id"] == "call-selector"

    wrong = dict(expected, prompt="different nonce-bearing prompt")
    with pytest.raises(AssertionError):
        assert_normalized_canary_call(
            {"tool_calls": [_canonical_canary_call("call-wrong", wrong)]},
            tools,
            required,
        )
    duplicate = _canonical_canary_call("call-duplicate", expected)
    with pytest.raises(AssertionError):
        assert_normalized_canary_call(
            {"tool_calls": [duplicate, copy.deepcopy(duplicate)]},
            tools,
            required,
        )


def test_production_custom_none_text_is_semantic_success(monkeypatch):
    model = unique_openai_direct_defaults()[0]
    tools = full_registry_canary_tools()
    candidate = _bind_candidate(
        model,
        tools,
        [{"role": "user", "content": "Use the supplied tool result."}],
        "none",
    )
    resolved_model = model.split("::", 1)[-1]
    source = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": "Use the supplied tool result."}],
        "reasoning_effort": "medium",
        "max_completion_tokens": CANARY_MAX_TOKENS,
        "tools": copy.deepcopy(tools),
        "tool_choice": "none",
    }
    target = {
        "provider": "openai",
        "resolved_model": resolved_model,
        "usage_model": normalize_model_identity(model),
        "base_url": "https://api.openai.com/v1",
    }
    assert candidate.forbids_tool_call is True
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"role": "assistant", "content": "semantic text"},
        normalized_usage={},
    )
    assert observation.semantic_kind == "chat_message"

    import ouroboros.openai_chat_dispatch as dispatch
    import ouroboros.request_wire_recovery as recovery

    attempt_id = "full-registry-custom-none-text"
    capture = PhysicalAttemptCapture(
        attempt_id=attempt_id,
        model=candidate.physical_model,
        provider="openai",
        state="settled",
        candidate_measurement_kind="canonical_json_v1",
        candidate_raw_sha256=candidate.candidate_sha256,
        candidate_manifest_ref={
            "path": f"physical/{attempt_id}.json",
            "call_id": attempt_id,
            "sha256": canonical_sha256(attempt_id),
        },
        provider_status_code=200,
    )
    issued = []
    real_bind = recovery.bind_wire_compatibility_receipt

    def record_receipt(**kwargs):
        receipt = real_bind(**kwargs)
        issued.append(receipt)
        return receipt

    monkeypatch.setattr(recovery, "bind_wire_compatibility_receipt", record_receipt)
    with recovery.request_wire_call_scope():
        recovery.register_wire_candidate(candidate, source_payload=source, target=target)
        recovery.note_wire_send_succeeded(capture)
        message, usage = dispatch.normalize_direct_openai_completion(
            {"role": "assistant", "content": "semantic text"},
            {
                "provider": "openai",
                "resolved_model": normalize_model_identity(model),
                "prompt_tokens": 120,
                "completion_tokens": 12,
            },
            None,
        )
        recovery.finalize_wire_response(message, usage)

    assert message == {"role": "assistant", "content": "semantic text"}
    assert dispatch.CUSTOM_RECEIPTS_USAGE_KEY not in usage
    assert len(issued) == 1
    assert issued[0].semantic_kind == "chat_message"
    assert_openai_canary_usage(usage, model)


def test_public_chat_builds_full_registry_request_for_every_matrix_row():
    tools = full_registry_canary_tools()
    names = [tool["function"]["name"] for tool in tools]
    for canary in provider_canary_matrix():
        nonce = f"unit-{canary.canary_id}"
        expected = delegate_start_canary_arguments(nonce)

        class FakeClient:
            def __init__(self):
                self.calls = []

            def chat(self, **kwargs):
                self.calls.append(copy.deepcopy(kwargs))
                ordinal = len(self.calls)
                if ordinal == 1:
                    calls = [
                        _canonical_canary_call(
                            f"call-{canary.canary_id}-1", expected,
                        )
                    ]
                    if canary.continue_to_final:
                        calls.append(_canonical_canary_call(
                            f"call-{canary.canary_id}-2", expected,
                        ))
                    return (
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": calls,
                        },
                        _fake_usage(canary, ordinal),
                    )
                return (
                    {
                        "role": "assistant",
                        "content": f"FULL_REGISTRY_CONTINUED_{nonce}",
                    },
                    _fake_usage(canary, ordinal),
                )

        client = FakeClient()
        run_provider_contract_canary(
            client,
            canary=canary,
            tools=tools,
            nonce=nonce,
        )

        assert len(client.calls) == 1 + int(canary.continue_to_final)
        first = client.calls[0]
        assert first["model"] == canary.model
        assert first["reasoning_effort"] == canary.reasoning_effort
        assert first["max_tokens"] == CANARY_MAX_TOKENS
        assert first["no_proxy"] is True
        assert first["bypass_response_cache"] is False
        assert first["timeout"] == CANARY_TIMEOUT_SEC
        assert [tool["function"]["name"] for tool in first["tools"]] == names
        if canary.named_tool_choice:
            assert first["tool_choice"] == {
                "type": "function", "function": {"name": CANARY_TOOL_NAME},
            }
        else:
            assert first["tool_choice"] == "auto"
        if canary.continue_to_final:
            assert "After its tool result" in first["messages"][0]["content"]
            assert "expected_final_marker" in first["messages"][0]["content"]
            second = client.calls[1]
            assert second["tool_choice"] == "none"
            assert second["bypass_response_cache"] is False
            assert second["max_tokens"] == CANARY_CONTINUATION_MAX_TOKENS
            assert [tool["function"]["name"] for tool in second["tools"]] == [
                CANARY_TOOL_NAME,
            ]
            assert [message["role"] for message in second["messages"]] == [
                "user", "assistant", "tool", "tool",
            ]
            assert [message["tool_call_id"] for message in second["messages"][2:]] == [
                f"call-{canary.canary_id}-1",
                f"call-{canary.canary_id}-2",
            ]
            assert all(nonce in message["content"] for message in second["messages"][2:])
        else:
            assert "After its tool result" not in first["messages"][0]["content"]


def test_canary_retries_one_semantic_empty_first_turn(monkeypatch):
    import tests.provider_contract_ci as contract

    monkeypatch.setattr(contract.time, "sleep", lambda _seconds: None)
    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")
    nonce = "empty-first-turn"
    expected = delegate_start_canary_arguments(nonce)

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if len(self.calls) == 1:
                return {}, {**_fake_usage(canary, 1), "completion_tokens": 0}
            return (
                {"tool_calls": [_canonical_canary_call("call-recovered", expected)]},
                _fake_usage(canary, 2),
            )

    client = FakeClient()
    run_provider_contract_canary(
        client, canary=canary, tools=full_registry_canary_tools(), nonce=nonce,
    )
    assert [call["bypass_response_cache"] for call in client.calls] == [False, True]
    assert {
        key: value for key, value in client.calls[0].items() if key != "bypass_response_cache"
    } == {
        key: value for key, value in client.calls[1].items() if key != "bypass_response_cache"
    }


def test_canary_retries_one_semantic_empty_continuation(monkeypatch):
    import tests.provider_contract_ci as contract

    monkeypatch.setattr(contract.time, "sleep", lambda _seconds: None)
    canary = next(row for row in provider_canary_matrix() if row.continue_to_final)
    nonce = "empty-continuation"
    expected = delegate_start_canary_arguments(nonce)

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if len(self.calls) == 1:
                return (
                    {"tool_calls": [_canonical_canary_call("call-first", expected)]},
                    _fake_usage(canary, 1),
                )
            if len(self.calls) == 2:
                return {}, {**_fake_usage(canary, 2), "completion_tokens": 0}
            return (
                {"content": f"FULL_REGISTRY_CONTINUED_{nonce}"},
                _fake_usage(canary, 3),
            )

    client = FakeClient()
    run_provider_contract_canary(
        client, canary=canary, tools=full_registry_canary_tools(), nonce=nonce,
    )
    assert [call["bypass_response_cache"] for call in client.calls] == [
        False, False, True,
    ]
    assert {
        key: value for key, value in client.calls[1].items() if key != "bypass_response_cache"
    } == {
        key: value for key, value in client.calls[2].items() if key != "bypass_response_cache"
    }


def test_canary_repeated_semantic_empty_stays_red(monkeypatch):
    import tests.provider_contract_ci as contract

    monkeypatch.setattr(contract.time, "sleep", lambda _seconds: None)
    canary = next(row for row in provider_canary_matrix() if row.canary_id == "openrouter_grok")

    class EmptyClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return {}, {**_fake_usage(canary, len(self.calls)), "completion_tokens": 0}

    client = EmptyClient()
    with pytest.raises(AssertionError) as caught:
        run_provider_contract_canary(
            client, canary=canary, tools=full_registry_canary_tools(), nonce="twice-empty",
        )
    diagnostic = caught.value.args[0]["semantic_empty_provider_response"]
    assert diagnostic["attempts"] == 2
    assert diagnostic["message_keys"] == []
    assert [call["bypass_response_cache"] for call in client.calls] == [False, True]


@pytest.mark.parametrize("canary_id", ["openrouter_grok", "openrouter_fable"])
def test_canary_permanent_empty_and_nonempty_malformed_do_not_retry(monkeypatch, canary_id):
    import tests.provider_contract_ci as contract

    monkeypatch.setattr(contract.time, "sleep", lambda _seconds: None)
    canary = next(row for row in provider_canary_matrix() if row.canary_id == canary_id)

    class PermanentClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            secret = "sk-or-v1-" + "A" * 40
            usage = {
                **_fake_usage(canary, 1),
                "completion_tokens": 0,
                "provider_error": {
                    "kind": "bad_request", "code": "400", "message": secret,
                },
            }
            return {}, usage

    permanent = PermanentClient()
    with pytest.raises(AssertionError) as caught:
        run_provider_contract_canary(
            permanent, canary=canary, tools=full_registry_canary_tools(), nonce="permanent",
        )
    assert len(permanent.calls) == 1
    assert "sk-or-v1-" not in str(caught.value)
    assert "***REDACTED***" in str(caught.value)

    class MalformedClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return {"content": "not a tool call"}, _fake_usage(canary, 1)

    malformed = MalformedClient()
    with pytest.raises(AssertionError):
        run_provider_contract_canary(
            malformed, canary=canary, tools=full_registry_canary_tools(), nonce="malformed",
        )
    assert len(malformed.calls) == 1
