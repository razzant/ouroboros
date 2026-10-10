"""Effort constraints and evidence through the existing physical send drivers."""
import asyncio
import copy

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.request_wire_contract import payload_effort
from ouroboros.request_wire_recovery import (
    plan_next_wire_retry,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
)
from tests.test_request_wire_recovery_phase2b import (
    _capture,
    _Rejected,
    _Response,
    _store,
    _target,
    _value_payload,
)
from tests.test_request_wire_recovery_phase2b import (
    evidence_root as _evidence_root,
)
from tests._usage_store_testing import ledger_rows

evidence_root = _evidence_root


@pytest.fixture(autouse=True)
def isolated_effort_disclosure():
    """Builder-only probes never leave a call's pending note for another test."""
    from ouroboros.llm_capability_policy import _EFFORT_CLAMP_CVAR

    token = _EFFORT_CLAMP_CVAR.set(None)
    try:
        yield
    finally:
        _EFFORT_CLAMP_CVAR.reset(token)


ENUM_MESSAGE = ('reasoning.effort: Invalid option: expected one of '
                '"none"|"minimal"|"low"|"medium"|"high"|"xhigh"|"max"')


def _error(message=ENUM_MESSAGE, **fields):
    return {"status": 400, "message": message, **fields}


def _exception(error):
    rejected = _Rejected(error["message"], error.get("status"))
    rejected.body = {"error": {key: value for key, value in error.items() if key != "status"}}
    return rejected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("requested,error,applied", [
    pytest.param("ultra", _error(), "max", id="downward"),
    pytest.param("none", _error(
        "Reasoning is mandatory and cannot be disabled", param="reasoning.effort",
        enum=["high", "medium"]), "medium", id="mandatory-structured"),
    pytest.param("none", _error(
        'reasoning.effort is mandatory: supported values are "high", "medium"'),
        "medium", id="mandatory-text"),
    pytest.param("low", _error(
        "Invalid option", param="reasoning.effort", allowed_values=["high", "medium"]),
        "medium", id="all-higher-structured"),
    pytest.param("low", _error('reasoning.effort: expected one of "high"|"medium"'),
        "medium", id="all-higher-text"),
    pytest.param("medium", _error(
        "Invalid option", param="reasoning.effort", allowed_values=["minimal", "ultra"]),
        "minimal", id="advertised-minimal-below"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. "
                                 "Supported values are: 'low', 'medium', or 'high'."),
        "high", id="serial-or-conjunction"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. "
                                 "Supported values are: 'low', 'medium', and 'high'."),
        "high", id="serial-and-conjunction"),
    pytest.param("ultra", _error("reasoning.effort: Input should be 'low', 'medium' or 'high' "
                                 "and xhigh requires a pro plan"),
        "high", id="conjunction-stops-at-prose"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. Supported values are "
                                 "low, medium, or high and xhigh requires a pro plan"),
        "high", id="bare-second-conjunction-is-prose"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. Supported values are "
                                 "'low', 'medium', or 'high' and 'xhigh' is not supported"),
        "high", id="quoted-second-conjunction-is-prose"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. Supported values are "
                                 "low, medium, high and xhigh is not supported"),
        "high", id="first-conjunction-negative-prose"),
    pytest.param("xhigh", _error("reasoning.effort value 'xhigh' is not supported. Supported values are "
                                 "low, medium, high and xhigh is not supported"),
        "high", id="negative-prose-does-not-suppress-retry"),
    pytest.param("xhigh", _error("reasoning.effort value 'xhigh' is not supported. Supported values are "
                                 "low, medium, high and xhigh is unsupported"),
        "high", id="copula-negative-prose-does-not-suppress-retry"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. Supported values are "
                                 "low, medium, high, xhigh requires a pro plan"),
        "high", id="bare-comma-negative-prose"),
    pytest.param("ultra", _error("reasoning.effort value 'ultra' is not supported. Supported values are "
                                 "low or medium or high"),
        "high", id="chained-conjunctions"),
    pytest.param("xhigh", _error("reasoning.effort value 'xhigh' is not supported. Supported values are "
                                 "low, medium, or high and xhigh requires a pro plan"),
        "high", id="prose-tier-does-not-suppress-resend"),
])
def test_enum_without_scalar_echo_reaches_driver_and_learns_only_after_success(
    evidence_root, tmp_path, asynchronous, body_error, requested, error, applied,
):
    target = {**_target(), "requested_reasoning_effort": requested}
    source = _value_payload("nested", target, requested)
    sent = []

    def send(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            if body_error:
                return _Response({"error": error, "choices": [], "usage": {}})
            raise _exception(error)
        assert _store(evidence_root) is None
        assert payload_effort(candidate) == applied
        return _Response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"cost": 0.125, "effort": {"requested": "low", "reported": "ultra"},
                      "effort_resolution": {"observed": "ultra"}},
        })

    async def send_async(**candidate):
        return send(**candidate)

    client = LLMClient(api_key="unused")

    async def call_async():
        response = await client._create_chat_completion_with_retries_async(send_async, source, target)
        return client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)

    with request_wire_call_scope(), ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="enum")):
        if asynchronous:
            _, usage = asyncio.run(call_async())
        else:
            response = client._create_chat_completion_with_retries(send, source, target)
            _, usage = client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)
    assert [payload_effort(p) for p in sent] == [requested, applied]
    assert all(p["extra_body"]["reasoning"]["exclude"] is False for p in sent)
    assert usage["effort"]["requested"] == requested
    assert usage["effort"]["sent"] == {"extra_body.reasoning": {"effort": applied, "exclude": False}}
    assert usage["effort"]["reported"] is None
    assert usage["request_wire"]["applied_effort_source"] == "sent_candidate"
    rows = ledger_rows(tmp_path)
    assert rows[-1]["state"] == "settled"
    assert rows[-1]["effort"] == usage["effort"]
    assert "effort_resolution" not in rows[-1] and "effort_resolution" not in usage
    assert rows[-1]["cost_usd"] == 0.125 and rows[-1]["cost_final"] is True
    assert _store(evidence_root)
    with request_wire_call_scope():
        assert payload_effort(prepare_wire_payload_for_send(target, source, api_surface="chat.completions")) == applied
        supported = _value_payload("nested", target, "high")
        assert payload_effort(prepare_wire_payload_for_send(target, supported, api_surface="chat.completions")) == "high"
        changed_route = {**target, "base_url": "https://different.invalid/v1"}
        assert payload_effort(prepare_wire_payload_for_send(changed_route, source, api_surface="chat.completions")) == requested


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("carrier,requested,error,applied", [
    pytest.param("nested", "none", _error("Reasoning is mandatory and cannot be disabled"),
                 "low", id="legacy-mandatory"),
    pytest.param("nested", "minimal", _error("Mandatory parameter", param="reasoning.effort"),
                 "low", id="mandatory-alias"),
    pytest.param("nested", "none", _error("Unsupported parameter", param="reasoning"),
                 "provider_default", id="default-alias"),
    pytest.param("nested", "none", _error("Unsupported parameter", param="extra_body.reasoning"),
                 "provider_default", id="default-nested"),
    pytest.param("top", "none", _error("Unsupported parameter", param="reasoning_effort"),
                 "provider_default", id="default-top"),
    pytest.param("nested", "low", _error(
        "reasoning.effort value 'low' is not supported; use 'medium' or 'high'"),
        None, id="arbitrary-quotes"),
    pytest.param("nested", "low", _error("Invalid option", param="reasoning.effort"),
                 None, id="unknown-minimum"),
    pytest.param("nested", "none", _error(
        "Reasoning is mandatory", param="reasoning.effort", enum=["future"]),
        None, id="unknown-enum"),
    pytest.param("nested", "none", _error(
        "Reasoning is mandatory", param="temperature", enum=["medium", "high"]),
        None, id="other-field"),
    pytest.param("nested", "none", None, None, id="success"),
])
def test_minimum_recovery_preserves_legacy_aliases_and_requires_positive_evidence(
    evidence_root, tmp_path, asynchronous, body_error, carrier, requested, error, applied,
):
    target = _target()
    source = _value_payload(carrier, target, requested)
    sent = []

    def send(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1 and error is not None:
            if body_error:
                return _Response({"error": error, "choices": [], "usage": {}})
            raise _exception(error)
        return _Response()

    async def send_async(**candidate):
        return send(**candidate)

    client = LLMClient(api_key="unused")

    async def call_async():
        response = await client._create_chat_completion_with_retries_async(send_async, source, target)
        if applied or error is None:
            client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)
        return response

    def call():
        if asynchronous:
            return asyncio.run(call_async())
        response = client._create_chat_completion_with_retries(send, source, target)
        if applied or error is None:
            client._normalize_remote_response(response.model_dump(), target, skip_cost_fetch=True)
        return response

    with request_wire_call_scope(), ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="minimum")), ua.physical_attempt_limit(2 if applied else 1):
        if error is not None and applied is None and not body_error:
            with pytest.raises(_Rejected):
                call()
        else:
            response = call()
            if error is not None and applied is None:
                assert response.model_dump()["error"] == error
    sent_effort = "" if applied == "provider_default" else applied
    assert [payload_effort(p) for p in sent] == ([requested, sent_effort] if applied else [requested])
    assert bool(_store(evidence_root)) is bool(applied)


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("carrier", ["top", "nested", "anthropic"])
@pytest.mark.parametrize("supported,applied", [
    # The old parser accepted the negative 'max' as the highest prescription.
    ("; 'max' is not supported. Supported values are: 'low', 'medium', 'high', 'xhigh'.", "xhigh"),
    # The old parser read the serial conjunction as a value 'or' and lost 'high'.
    (". Supported values are: 'low', 'medium', or 'high'.", "high"),
    # The old parser read a clause after the serial conjunction as one more value.
    (". Supported values are low, medium, or high and xhigh requires a pro plan", "high"),
])
def test_positive_constraints_are_bound_to_field_and_ignore_negative_quotes(evidence_root, carrier, body_error, supported, applied):
    target = _target("anthropic" if carrier == "anthropic" else "openrouter")
    field = {"top": "reasoning_effort", "nested": "reasoning.effort", "anthropic": "output_config.effort"}[carrier]
    source = _value_payload(carrier, target, "ultra")
    error = _error(f"{field} value 'ultra' is not supported{supported}")
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="messages" if carrier == "anthropic" else "chat.completions")
        retry = plan_next_wire_retry(sent, error=error if body_error else _exception(error), body_error=body_error)
        assert payload_effort(retry) == applied


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("carrier", ["top", "nested"])
@pytest.mark.parametrize("requested,error,applied", [
    pytest.param("medium", _error("Invalid option", param="{field}", enum=["minimal", "ultra"]),
                 "minimal", id="strongest-at-or-below-is-minimal"),
    pytest.param("medium", _error("Invalid option", param="{field}", enum=["minimal"]),
                 "minimal", id="only-minimal"),
    pytest.param("medium", _error('{field}: expected one of "minimal"|"ultra"'),
                 "minimal", id="text-minimal"),
    pytest.param("none", _error("Reasoning is mandatory", param="{field}", enum=["minimal", "high"]),
                 "minimal", id="mandatory-floor-minimal"),
    pytest.param("none", _error('{field} is mandatory: supported values are "high", "medium", or "minimal"'),
                 "minimal", id="mandatory-text-conjunction-minimal"),
    pytest.param("medium", _error("Invalid option", param="{field}", enum=["none", "high"]),
                 "high", id="none-is-not-comparable"),
    pytest.param("medium", _error("{field} value 'medium' is not supported; 'minimal' is not "
                                  "supported. Supported values are: 'ultra'"),
                 "ultra", id="negative-minimal-not-advertised"),
    pytest.param("minimal", _error("Invalid option", param="{field}", enum=["minimal", "high"]),
                 None, id="exact-minimal-accepted"),
    pytest.param("medium", _error("Invalid option", param="{field}", enum=["none"]),
                 None, id="only-none"),
    pytest.param("medium", _error("Invalid option", param="{field}", enum=["future", "turbo"]),
                 None, id="unknown-tiers"),
    pytest.param("medium", _error("Invalid option", param="temperature", enum=["minimal"]),
                 None, id="foreign-field"),
    pytest.param("low", _error("{field} value 'low' is not supported"),
                 None, id="unadvertised-walk-stops-at-low"),
])
def test_advertised_minimal_is_a_comparable_tier(evidence_root, carrier, body_error, requested, error, applied):
    field = {"top": "reasoning_effort", "nested": "reasoning.effort"}[carrier]
    error = {key: value.format(field=field) if isinstance(value, str) else value for key, value in error.items()}
    target = _target()
    source = _value_payload(carrier, target, requested)
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        failure = error if body_error else _exception(error)
        if not body_error:
            failure.physical_attempt_capture = _capture(sent, target)
        retry = plan_next_wire_retry(sent, error=failure, body_error=body_error)
    assert (payload_effort(retry) if retry is not None else None) == applied
    assert _store(evidence_root) is None  # Planning alone never learns a contract.


@pytest.mark.parametrize("error", [
    _error(ENUM_MESSAGE, param="temperature"),
    _error(ENUM_MESSAGE, code="insufficient_quota"),
    _error(ENUM_MESSAGE, code="context_length_exceeded"),
    _error(ENUM_MESSAGE, status=None),
    _error(ENUM_MESSAGE, status=429),
    _error("reasoning.effort: invalid option", param="reasoning.effort", code="invalid_option"),
    _error("reasoning.effort value 'ultra' invalid; temperature expected one of 'low', 'high'"),
    _error("reasoning.effort value 'high' invalid; expected one of 'low', 'medium'"),
    _error("reasoning.effort value 'ultra' invalid; temperature expected one of 'low', or 'high'"),
    _error("reasoning.effort value 'high' invalid; supported values are 'low', 'medium', or 'xhigh'"),
    _error(ENUM_MESSAGE, param="reasoning.effort", value="high"),
])
@pytest.mark.parametrize("body_error", [False, True])
def test_other_fields_unknown_and_noncompatibility_errors_do_not_remove_effort(evidence_root, error, body_error):
    target = _target()
    source = _value_payload("nested", target, "ultra")
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        assert plan_next_wire_retry(sent, error=error if body_error else _exception(error), body_error=body_error) is None
    assert payload_effort(source) == "ultra"
    assert _store(evidence_root) is None


def test_structured_enum_and_parameter_survive_sdk_exception_projection(evidence_root):
    target = _target()
    source = _value_payload("nested", target, "ultra")
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_next_wire_retry(sent, error=_exception(_error(
            "Invalid option", param="reasoning.effort", code="invalid_option", allowed_values=["low", "high"])))
    assert payload_effort(retry) == "high"


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("message,allowed", [
    ('reasoning_effort: expected one of "and"|"or"', ("and", "or")),
    ("reasoning_effort: allowed values are 'low', 'or', or 'high'", ("low", "or", "high")),
    ("reasoning_effort: allowed values are low, medium, or high", ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low' or higher", ("low",)),
    ("reasoning_effort: allowed values are 'or' and 'and'", ("or", "and")),
    ("reasoning_effort: allowed values are low, medium, or high and xhigh requires a pro plan",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low', 'medium', or 'high' and 'xhigh' is not supported",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low', 'medium', 'high', xhigh requires a pro plan",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are low, medium, high and xhigh is not supported",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are low, medium, high and xhigh is unsupported",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low', 'medium', 'high', and 'xhigh' isn't supported",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low', 'medium', 'high', 'xhigh' (unsupported on this plan)",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are low, medium, high, xhigh requires a pro plan",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are low or medium or high",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'high' (not recommended)", ("high",)),
    ("reasoning_effort: allowed values are low, medium, or high",
     ("low", "medium", "high")),
    ("reasoning_effort: allowed values are 'low', 'medium', or 'high' and 'xhigh' is not supported",
     ("low", "medium", "high")),
    ('reasoning_effort: expected one of "low"|"medium"|"high" and "xhigh" requires a pro plan',
     ("low", "medium", "high")),
])
def test_enum_conjunction_keeps_quoted_literals_and_stops_at_prose(body_error, message, allowed):
    from ouroboros.request_wire_recovery import _wire_rejection

    error = _error(message)
    assert _wire_rejection(error if body_error else _exception(error)).allowed == allowed


def test_rejection_from_another_physical_candidate_cannot_change_this_one(evidence_root):
    from ouroboros.request_wire_recovery import plan_wire_retry_from_exception

    target = _target()
    source = _value_payload("nested", target, "ultra")
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        error = _exception(_error())
        error.physical_attempt_capture = {}
        assert plan_next_wire_retry(sent, error=error) is None
        assert plan_wire_retry_from_exception(error) is None
        error.physical_attempt_capture = _capture({**sent, "model": "other"}, target)
        assert plan_next_wire_retry(sent, error=error) is None


@pytest.mark.parametrize("body_error", [False, True])
def test_carrierless_repair_requires_its_prepared_logical_and_physical_input(evidence_root, body_error):
    from ouroboros.llm_attempt import _finalized_physical_candidate
    from ouroboros.send_clock import MainSendClock, SendClockPolicy

    target = _target()
    source = {"model": target["resolved_model"], "messages": [{"role": "user", "content": "hi"}],
              "temperature": 0.7, "timeout": 123}
    rejection = _error("Unsupported parameter: temperature", param="temperature")
    error = rejection if body_error else _exception(rejection)
    with request_wire_call_scope(), MainSendClock(SendClockPolicy("UTC")).bound():
        sent = _finalized_physical_candidate(target, source, "chat.completions", fresh_clock=True)
        if not body_error:
            error.physical_attempt_capture = {}
            assert plan_next_wire_retry(source, error=error) is None
            error.physical_attempt_capture = _capture({**sent, "model": "foreign"}, target)
            assert plan_next_wire_retry(source, error=error) is None
            error.physical_attempt_capture = _capture(sent, target)
        assert plan_next_wire_retry({**source, "temperature": 0.8}, error=error, body_error=body_error) is None
        retry = plan_next_wire_retry(source, error=error, body_error=body_error)
        assert retry == {key: value for key, value in sent.items() if key != "temperature"}
    assert _store(evidence_root) is None  # This path never teaches a durable contract.


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("error,omitted", [
    (_error("thinking.type: Input should be 'adaptive'"), True),
    (_error("thinking.type: Input should be 'enabled', 'adaptive', or 'disabled'"), False),
    (_error("Invalid type", param="thinking.type", enum=["adaptive"], value="disabled"), True),
    (_error("Invalid type", param="thinking.type", enum=["adaptive", "disabled"]), False),
    (_error("Invalid type", param="thinking.type", enum=["adaptive"], value="enabled"), False),
    (_error("thinking.type: Input should be 'adaptive'", param="output_config.effort"), False),
    (_error("thinking value 'none' is unsupported", param="thinking"), False),
])
def test_disabled_carrier_type_refusal_is_not_an_individual_tier_refusal(evidence_root, body_error, error, omitted):
    target = _target("anthropic", "future")
    source = {"model": "future", "messages": [{"role": "user", "content": "hi"}],
              "thinking": {"type": "disabled"}, "max_tokens": 128}
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="messages")
        failure = error if body_error else _exception(error)
        if not body_error:
            failure.physical_attempt_capture = _capture(sent, target)
        retry = plan_next_wire_retry(source, error=failure, body_error=body_error)
    assert retry == ({key: value for key, value in sent.items() if key != "thinking"} if omitted else None)


@pytest.mark.parametrize("body_error", [False, True])
def test_logical_input_binding_survives_preparation_but_rejects_changed_input(evidence_root, body_error):
    from ouroboros.llm_attempt import _finalized_physical_candidate
    from ouroboros.send_clock import MainSendClock, SendClockPolicy

    target = _target()
    source = _value_payload("nested", target, "ultra")
    source["timeout"] = 123
    source["messages"][0]["_context_capsule"] = {"kind": "host-only"}
    error = _error() if body_error else _exception(_error())
    with request_wire_call_scope(), MainSendClock(SendClockPolicy("UTC")).bound():
        sent = _finalized_physical_candidate(target, source, "chat.completions", fresh_clock=True)
        assert "timeout" not in sent
        assert "_context_capsule" not in sent["messages"][0]
        assert plan_next_wire_retry({**source, "model": "other"}, error=error, body_error=body_error) is None
        retry = plan_next_wire_retry(source, error=error, body_error=body_error)
        assert payload_effort(retry) == "max"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("nano", [False, True])
def test_real_driver_recovery_binds_timeout_capsule_and_clock(evidence_root, tmp_path, asynchronous, body_error, nano):
    from ouroboros.send_clock import MainSendClock, SendClockPolicy

    target = _target()
    source = _value_payload("nested", target, "ultra")
    if nano:
        source["max_tokens"] = 200000  # Forces real allowance reduction before sealing (the 128K window below).
    source["timeout"] = 123
    source["messages"][0]["_context_capsule"] = {"kind": "host-only"}
    original, sent = copy.deepcopy(source), []
    # A rendered Nano reaches the send only through the bound Main context (no keyword on the call chain).
    physical = ua.PhysicalAttemptContext(
        profile="owner_nano", rendered_mode="nano", measurement_basis="cold_estimate", route_fp="r", round_id="x:round:1",
        target_total_tokens=85_000, capacity_total_tokens=128_000, context_target_miss=False, automatic_pass_used=False,
    ) if nano else None

    def send(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            if body_error:
                return _Response({"error": _error(), "choices": [], "usage": {}})
            raise _exception(_error())
        return _Response()

    client = LLMClient(api_key="unused")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="binding")), MainSendClock(SendClockPolicy("UTC")).bound(), \
            ua.bind_physical_attempt_context(physical):
        if asynchronous:
            async def async_send(**candidate):
                return send(**candidate)
            asyncio.run(client._create_chat_completion_with_retries_async(async_send, source, target))
        else:
            client._create_chat_completion_with_retries(send, source, target)
    assert source == original
    assert [payload_effort(candidate) for candidate in sent] == ["ultra", "max"]
    assert all(candidate["timeout"] == 123 for candidate in sent)
    assert all("_context_capsule" not in candidate["messages"][0] for candidate in sent)
    if nano:
        assert all(8_192 <= candidate["max_tokens"] < original["max_tokens"] for candidate in sent)
        assert len({candidate["max_tokens"] for candidate in sent}) == 1  # every rung: one rule on one source


@pytest.mark.parametrize("control", ["budget", "stop"])
@pytest.mark.parametrize("requested,error", [
    ("ultra", _error()),
    ("none", _error("Reasoning is mandatory", param="reasoning.effort", enum=["medium", "high"])),
    ("low", _error("Invalid option", param="reasoning.effort", enum=["medium", "high"])),
])
def test_reactive_effort_repair_never_adds_a_send_allowance(evidence_root, tmp_path, monkeypatch, control, requested, error):
    from ouroboros import llm_attempt

    target = _target()
    source = _value_payload("nested", target, requested)
    sent = []

    def send(**candidate):
        sent.append(candidate)
        if control == "stop":
            def stopped():
                raise llm_attempt.PhysicalDispatchInterrupted("stop")
            monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", stopped)
        raise _exception(error)

    failure = ua.PhysicalAttemptLimitExceeded if control == "budget" else llm_attempt.PhysicalDispatchInterrupted
    with request_wire_call_scope(), ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="limit")), ua.physical_attempt_limit(1), pytest.raises(failure):
        LLMClient(api_key="unused")._create_chat_completion_with_retries(send, source, target)
    assert len(sent) == 1
    assert _store(evidence_root) is None


@pytest.mark.parametrize("provider,requested,sent_effort", [("deepseek", "ultra", "max"), ("zai", "medium", "high"), ("anthropic", "minimal", "low")])
def test_original_preference_survives_native_mapping_in_usage_and_ledger(evidence_root, tmp_path, monkeypatch, provider, requested, sent_effort):
    from ouroboros.llm_attempt import (
        _attempt_request,
        _candidate_before_dispatch,
        _execute_candidate,
        attach_processing_receipt,
    )

    target = _target(provider, "future")
    client = LLMClient(api_key="unused")
    source = client._build_remote_candidate(target, [{"role": "user", "content": "hello"}], requested, 32, "auto", None, None)
    assert payload_effort(source) == sent_effort
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="projection")):
        request = _attempt_request(target, source)
        _execute_candidate(request, lambda: _Response(), _candidate_before_dispatch(source, request))
        usage = {}
        attach_processing_receipt(target, usage)
    assert usage["effort"]["requested"] == requested
    assert usage["effort"]["reported"] is None
    rows = ledger_rows(tmp_path)  # one row per attempt: the settled row carries the candidate effort
    assert [row["state"] for row in rows] == ["settled"]
    assert rows[-1]["effort"] == usage["effort"]


def test_switch_model_next_round_keeps_preference_after_native_projection(tmp_path, monkeypatch):
    from ouroboros.loop import _apply_runtime_overrides
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.tools.registry import ToolRegistry

    ctx = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)._ctx
    monkeypatch.setattr(LLMClient, "available_models", lambda *_: ["zai::future", "openai::future"])
    assert "next round" in _switch_model(ctx, model="zai::future", effort="ultra")
    model, local, effort = _apply_runtime_overrides(ctx, "openai::future", False, "medium")
    assert (model, local, effort) == ("zai::future", False, "ultra")
    target = _target("zai", "future")
    payload = LLMClient(api_key="unused")._build_remote_candidate(target, [], effort, 32, "auto", None, None)
    assert payload_effort(payload) == "max"
    assert _apply_runtime_overrides(ctx, model, local, effort) == (model, local, "ultra")
    assert "next round" in _switch_model(ctx, model="openai::future")
    assert _apply_runtime_overrides(ctx, model, local, effort) == ("openai::future", False, "ultra")


def test_native_omission_is_recorded_without_inventing_provider_effort(tmp_path, monkeypatch):
    from types import SimpleNamespace

    sent = []
    completion = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="done", function_call=None), finish_reason="stop")])
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_get_gigachat_client", lambda *a, **kw: SimpleNamespace(
        chat=lambda payload: sent.append(payload) or completion))
    target = _target("gigachat", "GigaChat")
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="omitted")):
        _, usage = client._chat_gigachat(target, [], None, "ultra", 32, "auto")
    assert len(sent) == 1 and "reasoning_effort" not in sent[0]
    assert usage["effort"] == {
        "requested": "ultra", "sent": {}, "sent_state": "omitted", "sent_source": "host_candidate",
        "reported": None, "report_source": None,
    }
    rows = ledger_rows(tmp_path)
    assert rows[-1]["state"] == "settled"
    assert rows[-1]["effort"] == usage["effort"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_driver_retains_known_intent_with_omitted_effort(tmp_path, monkeypatch, asynchronous):
    from types import SimpleNamespace
    from ouroboros import llm_local, local_model

    sent = []
    client = LLMClient(api_key="unused")
    manager = SimpleNamespace(serving_context_evidence=lambda: {})
    monkeypatch.setattr(local_model, "get_manager", lambda: manager)
    monkeypatch.setattr(llm_local, "local_context_limits", lambda maximum: (0, maximum))
    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **payload: sent.append(payload) or _Response()))))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="local-intent")):
        kwargs = dict(messages=[{"role": "user", "content": "hi"}], model="local", use_local=True,
                      reasoning_effort="ultra", max_tokens=32)
        _, usage = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)
    assert len(sent) == 1 and "reasoning_effort" not in sent[0] and "reasoning" not in sent[0]
    assert usage["effort"] == {
        "requested": "ultra", "sent": {}, "sent_state": "omitted", "sent_source": "host_candidate",
        "reported": None, "report_source": None,
    }
    rows = ledger_rows(tmp_path)
    assert rows[-1]["state"] == "settled" and rows[-1]["effort"] == usage["effort"]


def test_provider_usage_cannot_supply_host_effort_evidence(monkeypatch):
    from ouroboros import llm_attempt

    monkeypatch.setattr(llm_attempt, "last_physical_attempt_capture", lambda: None)
    usage = {"effort": {"reported": "ultra", "sent_source": "host_candidate"},
             "effort_resolution": {"observed": "ultra"}}
    llm_attempt.attach_processing_receipt(_target(), usage)
    assert "effort" not in usage
    assert "effort_resolution" not in usage


def test_late_receipt_keeps_candidate_effort_without_inheriting_an_older_report(tmp_path):
    from ouroboros.llm_attempt import _attempt_request

    target = {**_target(), "requested_reasoning_effort": "ultra"}
    request = _attempt_request(target, _value_payload("nested", target, "max"))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="late")):
        reservation = ua.reserve_attempt(request)
        ua.mark_dispatched(reservation)
        # A historical administrative row may contain an older observation.
        ua._transition(reservation, "settled", settle_reason="abandoned", cost_usd=None, cost_final=False,
                       effort={**request.effort, "reported": "high", "report_source": "provider_response"})
        assert ledger_rows(tmp_path)[-1]["effort"]["reported"] == "high"
        ua.settle_attempt(reservation, {}, cost_usd=0.25, cost_final=True)
    rows = ledger_rows(tmp_path)
    assert rows[-1]["effort"] == request.effort
    assert rows[-1]["settle_reason"] == "late_receipt"
    assert rows[-1]["cost_usd"] == 0.25 and rows[-1]["cost_final"] is True
