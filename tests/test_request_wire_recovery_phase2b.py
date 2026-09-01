"""Production-driver regressions for exact-route request-wire recovery."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros.llm import LLMClient, add_usage
from ouroboros.request_wire_contract import (
    canonical_sha256,
    infer_tool_dialect,
    payload_effort,
    physical_candidate_sha256,
)
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    finalize_wire_response,
    merge_request_wire_usage,
    note_wire_send_failed,
    note_wire_send_succeeded,
    plan_next_wire_retry,
    plan_wire_retry_from_body_error,
    plan_wire_retry_from_exception,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
    request_wire_disclosures,
)
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture,
    PhysicalAttemptLimitExceeded,
    UsageScope,
    physical_attempt_limit,
    usage_projection,
    usage_scope,
)


class _Rejected(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "type": "invalid_request_error"}}


class _Response:
    def __init__(self, body=None):
        self._body = body or {
            "id": "resp-ok",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        }

    def model_dump(self):
        return copy.deepcopy(self._body)


def _target(provider="openrouter", model="vendor/future"):
    host = "openrouter.example" if provider == "openrouter" else f"{provider}.example"
    return {
        "provider": provider,
        "resolved_model": model,
        "usage_model": f"{provider}/{model}",
        "base_url": f"https://{host}/v1",
        "contract_headers": {"api-version": "test"},
        "supports_openrouter_extensions": provider == "openrouter",
        "supports_generation_cost": False,
    }


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _payload(*, effort="high", toolful=True):
    payload = {
        "model": "vendor/future",
        "messages": [{"role": "user", "content": "probe"}],
        "reasoning_effort": effort,
        "temperature": 0.4,
        "max_tokens": 32,
    }
    if toolful:
        payload.update(tools=_tools(), tool_choice="auto")
    return payload


def _capture(payload, target, attempt="attempt-wire"):
    return PhysicalAttemptCapture(
        attempt_id=attempt,
        model=target["usage_model"],
        provider=target["provider"],
        state="settled",
        candidate_measurement_kind="canonical_json_v1",
        candidate_raw_sha256=physical_candidate_sha256(payload),
        candidate_manifest_ref={
            "path": "/private/candidate.json",
            "call_id": attempt,
            "sha256": canonical_sha256("manifest"),
        },
    )


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    return tmp_path


def _store(root):
    path = root / "state" / wire.REQUEST_WIRE_STATE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def test_sequential_constraints_compose_and_remain_source_predicated(evidence_root):
    target = _target()
    source = _payload()
    with request_wire_call_scope():
        first = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        assert first == source
        note_wire_send_failed()
        second = plan_wire_retry_from_exception(
            _Rejected("reasoning_effort value 'high' is not supported")
        )
        assert second["reasoning_effort"] == "medium"
        assert second["temperature"] == 0.4
        note_wire_send_failed()
        third = plan_wire_retry_from_exception(_Rejected("temperature is unsupported"))
        assert third["reasoning_effort"] == "medium"
        assert "temperature" not in third
        note_wire_send_succeeded(_capture(third, target))
        usage = {}
        finalize_wire_response({"role": "assistant", "content": "ok"}, usage)
        assert len(usage["request_wire"]["applied_actions"]) == 2

    records = [record for entry in _store(evidence_root)["profiles"].values()
               for record in entry["records"]]
    assert {record["action"]["kind"] for record in records} == {"set_value", "drop_field"}

    with request_wire_call_scope():
        learned = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
    assert learned["reasoning_effort"] == "medium"
    assert "temperature" not in learned

    low = _payload(effort="low")
    with request_wire_call_scope():
        untouched = prepare_wire_payload_for_send(target, low, api_surface="chat.completions")
    assert untouched["reasoning_effort"] == "low"
    assert "temperature" not in untouched


def test_exact_route_tool_shape_isolation(evidence_root):
    source = _payload(effort="medium")
    openrouter = _target()
    with request_wire_call_scope():
        prepare_wire_payload_for_send(openrouter, source, api_surface="chat.completions")
        repaired = plan_wire_retry_from_exception(_Rejected("temperature not supported"))
        note_wire_send_succeeded(_capture(repaired, openrouter))
        finalize_wire_response({"content": "ok"}, {})

    with request_wire_call_scope():
        same = prepare_wire_payload_for_send(openrouter, source, api_surface="chat.completions")
    assert "temperature" not in same

    direct = _target(provider="openai")
    with request_wire_call_scope():
        direct_payload = prepare_wire_payload_for_send(
            direct, source, api_surface="chat.completions"
        )
    assert direct_payload["temperature"] == 0.4

    toolless = _payload(effort="medium", toolful=False)
    with request_wire_call_scope():
        toolless_payload = prepare_wire_payload_for_send(
            openrouter, toolless, api_surface="chat.completions"
        )
    assert toolless_payload["temperature"] == 0.4

    openrouter_anthropic = _target(model="anthropic/claude-future")
    openrouter_anthropic_source = _payload(effort="medium")
    openrouter_anthropic_source["model"] = openrouter_anthropic["resolved_model"]
    with request_wire_call_scope():
        openrouter_anthropic_payload = prepare_wire_payload_for_send(
            openrouter_anthropic,
            openrouter_anthropic_source,
            api_surface="chat.completions",
        )
    assert openrouter_anthropic_payload["temperature"] == 0.4

    anthropic = _target(provider="anthropic")
    anthropic["resolved_model"] = anthropic["usage_model"] = "claude-future"
    native = {
        "model": "claude-future",
        "messages": [{"role": "user", "content": "probe"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "temperature": 0.4,
    }
    with request_wire_call_scope():
        native_payload = prepare_wire_payload_for_send(
            anthropic, native, api_surface="messages"
        )
    assert native_payload["temperature"] == 0.4


def test_unrepresentable_custom_catalog_falls_to_function_rung(evidence_root):
    """E4: a catalog the custom dialect cannot represent (CustomToolProjectionError)
    must fall to the function rung WITH a registered candidate. It used to be
    swallowed as a malformed payload: the raw payload went on the wire with
    state.current cleared, so every retry rung returned None and the turn died
    on the raw provider error."""
    huge = {f"field_{index:03d}": {"type": "string"} for index in range(300)}
    source = _payload()
    source["model"] = "gpt-future"
    source["tools"] = [{
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {"type": "object", "properties": huge},
        },
    }]
    target = _target(provider="openai", model="gpt-future")
    with request_wire_call_scope():
        sent = prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        registered = current_wire_candidate()
        # Never on the wire with a severed ladder: the function-dialect rung is
        # bound and registered, and the physical payload IS that candidate.
        assert registered is not None
        assert infer_tool_dialect(sent) == "function"
        assert registered.candidate_sha256 == physical_candidate_sha256(sent)
        note_wire_send_failed()
        # The retry ladder stays reachable after the projection failure.
        retry = plan_wire_retry_from_exception(_Rejected("temperature is unsupported"))
        assert retry is not None
        assert "temperature" not in retry


def _value_payload(carrier, target, effort):
    payload = {
        "model": target["resolved_model"],
        "messages": [{"role": "user", "content": "probe"}],
        "temperature": 0.4,
        "max_tokens": 32,
    }
    if carrier == "top":
        payload["reasoning_effort"] = effort
    elif carrier == "nested":
        payload["extra_body"] = {"reasoning": {"effort": effort, "exclude": False}}
    else:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
    return payload


@pytest.mark.parametrize("body_error", [False, True])
@pytest.mark.parametrize("carrier", ["top", "nested", "anthropic"])
def test_named_exact_effort_value_is_source_predicated_across_carriers(
    evidence_root, carrier, body_error,
):
    target = (
        _target(provider="anthropic", model="claude-future")
        if carrier == "anthropic" else
        _target(provider="openai" if carrier == "top" else "openrouter")
    )
    surface = "messages" if carrier == "anthropic" else "chat.completions"
    field = {
        "top": "reasoning_effort",
        "nested": "reasoning.effort",
        "anthropic": "output_config.effort",
    }[carrier]
    message = f"{field} value 'high' is not supported"
    source = _value_payload(carrier, target, "high")

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface=surface)
        retry = (
            plan_wire_retry_from_body_error({"status": 400, "message": message})
            if body_error else
            plan_wire_retry_from_exception(_Rejected(message))
        )
        assert retry is not None and payload_effort(retry) == "medium"
        note_wire_send_succeeded(_capture(retry, target, f"{carrier}-{body_error}"))
        finalize_wire_response({"content": "ok"}, {})

    low = _value_payload(carrier, target, "low")
    with request_wire_call_scope():
        prepared_low = prepare_wire_payload_for_send(target, low, api_surface=surface)
    assert payload_effort(prepared_low) == "low"
    if carrier == "anthropic":
        assert prepared_low["thinking"] == {"type": "adaptive"}


def test_ambiguous_named_effort_fails_open_but_explicit_carrier_absence_drops(
    evidence_root,
):
    target = _target(provider="openai")
    source = _payload()
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        assert plan_wire_retry_from_exception(
            _Rejected("reasoning_effort high is not supported")
        ) is None
        assert plan_wire_retry_from_exception(
            _Rejected("invalid value for reasoning_effort")
        ) is None

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_exception(
            _Rejected("reasoning_effort parameter is not supported")
        )
    assert retry is not None and "reasoning_effort" not in retry

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        assert plan_wire_retry_from_exception(
            _Rejected("temperature value '0.4' is not supported")
        ) is None
        assert plan_wire_retry_from_exception(
            _Rejected("temperature must be between 0 and 2")
        ) is None

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_exception(
            _Rejected("temperature parameter is not supported")
        )
    assert retry is not None and "temperature" not in retry

    no_effort = {"model": "future", "messages": [], "temperature": 0.4}
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, no_effort, api_surface="chat.completions")
        assert plan_next_wire_retry(
            no_effort,
            error=_Rejected("temperature value '0.4' is not supported"),
        ) is None
        assert plan_next_wire_retry(
            no_effort,
            error={
                "status": 400,
                "message": "temperature value '0.4' is not supported",
            },
            body_error=True,
        ) is None


def test_prescribed_lower_effort_tier_wins_over_the_one_rung_walk(evidence_root):
    target = _target(provider="openai")
    ultra = _payload(effort="ultra")
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, ultra, api_surface="chat.completions")
        prescribed = plan_wire_retry_from_exception(_Rejected(
            "reasoning_effort value 'ultra' is not supported. "
            "Supported values are: 'low', 'medium', 'high', 'xhigh'."
        ))
    assert prescribed is not None and payload_effort(prescribed) == "xhigh"

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, ultra, api_surface="chat.completions")
        bare = plan_wire_retry_from_exception(
            _Rejected("reasoning_effort value 'ultra' is not supported")
        )
    assert bare is not None and payload_effort(bare) == "max"

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, _payload(), api_surface="chat.completions")
        upward = plan_wire_retry_from_exception(
            _Rejected("reasoning_effort value 'high' is not supported; use 'max' or 'ultra'")
        )
    assert upward is not None and payload_effort(upward) == "medium"


def test_prescribed_jump_needs_a_quoted_tier_inside_the_retry_floor(evidence_root):
    target = _target(provider="openai")

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, _payload(effort="ultra"), api_surface="chat.completions")
        prose = plan_wire_retry_from_exception(_Rejected(
            "reasoning_effort value 'ultra' is not supported: too high for this model"
        ))
    assert prose is not None and payload_effort(prose) == "max"

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, _payload(effort="medium"), api_surface="chat.completions")
        english_word = plan_wire_retry_from_exception(_Rejected(
            "reasoning_effort value 'medium' is not supported; "
            "none of the selected endpoints accept it"
        ))
    assert english_word is not None and payload_effort(english_word) == "low"

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, _payload(effort="high"), api_surface="chat.completions")
        sub_low = plan_wire_retry_from_exception(_Rejected(
            "reasoning_effort value 'high' is not supported; use 'minimal' instead"
        ))
    assert sub_low is not None and payload_effort(sub_low) == "medium"


def test_failed_invalid_and_phase2a_finalized_attempts_never_poison(evidence_root):
    target = _target()
    source = _payload(effort="medium")
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_exception(_Rejected("temperature unsupported"))
        assert retry is not None
        note_wire_send_failed()
    assert _store(evidence_root) is None

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_exception(_Rejected("temperature unsupported"))
        note_wire_send_succeeded(_capture(retry, target, "attempt-body-error"))
        usage = {"provider_error": {"code": 400}}
        finalize_wire_response({"content": "looks valid"}, usage)
        assert usage["request_wire"]["candidate_sha256"] == physical_candidate_sha256(retry)
    assert _store(evidence_root) is None

    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, source, api_surface="chat.completions")
        retry = plan_wire_retry_from_exception(_Rejected("temperature unsupported"))
        note_wire_send_succeeded(_capture(retry, target, "attempt-phase2a"))
        phase2a = {"candidate_sha256": "phase2a-owned"}
        custom_receipts = (object(),)
        usage = {
            "request_wire": phase2a,
            "_request_wire_custom_receipts": custom_receipts,
        }
        finalize_wire_response({"content": "ok"}, usage)
        assert usage["request_wire"] is phase2a
        assert usage["_request_wire_custom_receipts"] is custom_receipts
    assert _store(evidence_root) is None


def test_statusless_and_5xx_body_errors_do_not_authorize_recovery(evidence_root):
    target = _target()
    with request_wire_call_scope():
        prepare_wire_payload_for_send(target, _payload(), api_surface="chat.completions")
        assert plan_wire_retry_from_body_error({
            "message": "temperature unsupported",
        }) is None
        assert plan_wire_retry_from_body_error({
            "status": 503, "message": "temperature unsupported",
        }) is None
        assert plan_wire_retry_from_body_error({
            "status": 429, "message": "temperature unsupported",
        }) is None
        recovered = plan_wire_retry_from_body_error({
            "status_code": 400, "message": "temperature unsupported",
        })
        assert recovered is not None and "temperature" not in recovered


def test_malformed_durable_store_fails_open_without_rewrite(evidence_root):
    path = evidence_root / "state" / wire.REQUEST_WIRE_STATE_FILE
    path.parent.mkdir(parents=True)
    malformed = b'{"schema_version":999,"profiles":"future"}\n'
    path.write_bytes(malformed)
    source = _payload()
    with request_wire_call_scope():
        prepared = prepare_wire_payload_for_send(
            _target(), source, api_surface="chat.completions"
        )
    assert prepared == source
    assert path.read_bytes() == malformed


def test_sync_async_exception_and_body_error_driver_parity(evidence_root, tmp_path):
    target = _target(provider="openai")
    client = LLMClient(api_key="unused")
    monkey_payload = _payload(effort="high")

    def run_sync(body_error: bool):
        root = tmp_path / f"wire-sync-{body_error}"
        wire.canonical_wire_evidence_root = lambda: root
        sent = []

        def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            if len(sent) == 1:
                if body_error:
                    return _Response({
                        "choices": [],
                        "error": {"code": 400, "message": "reasoning_effort value 'high' is not supported"},
                        "usage": {},
                    })
                raise _Rejected("reasoning_effort value 'high' is not supported")
            return _Response()

        with request_wire_call_scope(), usage_scope(UsageScope(
            drive_root=tmp_path / f"sync-{body_error}", task_id="sync",
        )):
            response = client._create_chat_completion_with_retries(
                create, copy.deepcopy(monkey_payload), target
            )
            _, usage = client._normalize_remote_response(
                response.model_dump(), target, skip_cost_fetch=True
            )
        return sent, usage

    async def run_async(body_error: bool):
        root = tmp_path / f"wire-async-{body_error}"
        wire.canonical_wire_evidence_root = lambda: root
        sent = []

        async def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            if len(sent) == 1:
                if body_error:
                    return _Response({
                        "choices": [],
                        "error": {"code": 400, "message": "reasoning_effort value 'high' is not supported"},
                        "usage": {},
                    })
                raise _Rejected("reasoning_effort value 'high' is not supported")
            return _Response()

        with request_wire_call_scope(), usage_scope(UsageScope(
            drive_root=tmp_path / f"async-{body_error}", task_id="async",
        )):
            response = await client._create_chat_completion_with_retries_async(
                create, copy.deepcopy(monkey_payload), target
            )
            _, usage = client._normalize_remote_response(
                response.model_dump(), target, skip_cost_fetch=True
            )
        return sent, usage

    for body_error in (False, True):
        sync_sent, sync_usage = run_sync(body_error)
        async_sent, async_usage = asyncio.run(run_async(body_error))
        assert [item["reasoning_effort"] for item in sync_sent] == ["high", "medium"]
        assert async_sent == sync_sent
        assert sync_usage["request_wire"]["applied_effort"] == "medium"
        assert async_usage["request_wire"]["applied_effort"] == "medium"


@pytest.mark.parametrize("order", ["exception_body", "body_exception"])
def test_mixed_exception_body_constraints_reach_one_terminal_candidate(
    evidence_root, tmp_path, order,
):
    target = _target(provider="openai")
    source = _payload(effort="high")

    def outcome(index):
        if index == 0:
            return "exception_value" if order == "exception_body" else "body_value"
        if index == 1:
            return "body_temperature" if order == "exception_body" else "exception_temperature"
        return "success"

    def run_sync():
        root = tmp_path / f"mixed-sync-{order}"
        wire.canonical_wire_evidence_root = lambda: root
        sent = []

        def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            event = outcome(len(sent) - 1)
            if event.startswith("exception"):
                field = "reasoning_effort value 'high'" if "value" in event else "temperature"
                raise _Rejected(f"{field} is not supported")
            if event.startswith("body"):
                field = "reasoning_effort value 'high'" if "value" in event else "temperature"
                return _Response({
                    "choices": [],
                    "error": {"code": 400, "message": f"{field} is not supported"},
                    "usage": {},
                })
            return _Response()

        with request_wire_call_scope(), usage_scope(UsageScope(
            drive_root=tmp_path / f"mixed-sync-usage-{order}", task_id="sync",
        )):
            response = LLMClient(api_key="unused")._create_chat_completion_with_retries(
                create, copy.deepcopy(source), target,
            )
            _, usage = LLMClient(api_key="unused")._normalize_remote_response(
                response.model_dump(), target, skip_cost_fetch=True,
            )
        return sent, usage

    async def run_async():
        root = tmp_path / f"mixed-async-{order}"
        wire.canonical_wire_evidence_root = lambda: root
        sent = []

        async def create(**candidate):
            sent.append(copy.deepcopy(candidate))
            event = outcome(len(sent) - 1)
            if event.startswith("exception"):
                field = "reasoning_effort value 'high'" if "value" in event else "temperature"
                raise _Rejected(f"{field} is not supported")
            if event.startswith("body"):
                field = "reasoning_effort value 'high'" if "value" in event else "temperature"
                return _Response({
                    "choices": [],
                    "error": {"code": 400, "message": f"{field} is not supported"},
                    "usage": {},
                })
            return _Response()

        with request_wire_call_scope(), usage_scope(UsageScope(
            drive_root=tmp_path / f"mixed-async-usage-{order}", task_id="async",
        )):
            client = LLMClient(api_key="unused")
            response = await client._create_chat_completion_with_retries_async(
                create, copy.deepcopy(source), target,
            )
            _, usage = client._normalize_remote_response(
                response.model_dump(), target, skip_cost_fetch=True,
            )
        return sent, usage

    sync_sent, sync_usage = run_sync()
    async_sent, async_usage = asyncio.run(run_async())
    assert async_sent == sync_sent
    assert [payload_effort(item) for item in sync_sent] == ["high", "medium", "medium"]
    assert "temperature" in sync_sent[1] and "temperature" not in sync_sent[2]
    assert len(sync_usage["request_wire"]["applied_actions"]) == 2
    assert async_usage["request_wire"]["applied_actions"] == (
        sync_usage["request_wire"]["applied_actions"]
    )
    assert async_usage["request_wire"]["candidate_sha256"] == (
        sync_usage["request_wire"]["candidate_sha256"]
    )


def test_mixed_failure_rail_blocks_third_candidate_without_learning(evidence_root, tmp_path):
    target = _target(provider="openai")
    client = LLMClient(api_key="unused")
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            return _Response({
                "choices": [],
                "error": {
                    "code": 400,
                    "message": "reasoning_effort value 'high' is not supported",
                },
                "usage": {},
            })
        raise _Rejected("temperature is not supported")

    with request_wire_call_scope(), usage_scope(UsageScope(
        drive_root=tmp_path / "mixed-rail", task_id="rail",
    )), physical_attempt_limit(2), pytest.raises(PhysicalAttemptLimitExceeded):
        client._create_chat_completion_with_retries(create, _payload(), target)
    assert len(sent) == 2
    assert _store(evidence_root) is None


def test_physical_attempt_rail_wins_and_commits_nothing(evidence_root, tmp_path):
    target = _target(provider="openai")
    client = LLMClient(api_key="unused")
    sends = []

    def create(**candidate):
        sends.append(copy.deepcopy(candidate))
        message = (
            "reasoning_effort value 'high' is not supported"
            if len(sends) == 1 else "temperature unsupported"
        )
        raise _Rejected(message)

    with request_wire_call_scope(), usage_scope(UsageScope(
        drive_root=tmp_path / "rail", task_id="rail",
    )), physical_attempt_limit(2), pytest.raises(PhysicalAttemptLimitExceeded):
        client._create_chat_completion_with_retries(create, _payload(), target)
    assert len(sends) == 2
    assert _store(evidence_root) is None


def test_exact_openrouter_parameter_404_keeps_existing_zero_settlement(
    evidence_root, tmp_path,
):
    target = _target()
    client = LLMClient(api_key="unused")
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            raise _Rejected(
                "404 No endpoints found for requested parameter temperature",
                status=404,
            )
        return _Response()

    usage_root = tmp_path / "zero-settlement"
    with request_wire_call_scope(), usage_scope(UsageScope(
        drive_root=usage_root, task_id="zero",
    )):
        response = client._create_chat_completion_with_retries(
            create, _payload(effort="medium"), target
        )
        client._normalize_remote_response(
            response.model_dump(), target, skip_cost_fetch=True
        )
    assert len(sent) == 2 and "temperature" not in sent[1]
    projection = usage_projection(usage_root)
    assert projection["attempt_counts"] == {"settled": 2}
    assert _store(evidence_root) is not None


def test_mandatory_reasoning_floor_is_exact_and_success_confirmed(
    evidence_root, tmp_path,
):
    target = _target()
    client = LLMClient(api_key="unused")
    source = {
        "model": target["resolved_model"],
        "messages": [{"role": "user", "content": "probe"}],
        "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
        "max_tokens": 32,
    }
    sent = []

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            raise _Rejected(
                "Reasoning is mandatory for this endpoint and cannot be disabled"
            )
        return _Response()

    with request_wire_call_scope(), usage_scope(UsageScope(
        drive_root=tmp_path / "mandatory", task_id="mandatory",
    )):
        response = client._create_chat_completion_with_retries(create, source, target)
        _, usage = client._normalize_remote_response(
            response.model_dump(), target, skip_cost_fetch=True
        )
    assert [item["extra_body"]["reasoning"]["effort"] for item in sent] == [
        "none", "low",
    ]
    assert usage["request_wire"]["requested_effort"] == "none"
    assert usage["request_wire"]["applied_effort"] == "low"
    actions = [record["action"] for entry in _store(evidence_root)["profiles"].values()
               for record in entry["records"]]
    assert actions == [{
        "field": "effort",
        "from": "none",
        "kind": "set_value",
        "mode": "floor",
        "reason_code": "provider_required_reasoning",
        "to": "low",
    }]


def test_contextvars_are_task_local_and_usage_aggregates_nested_calls(evidence_root):
    async def worker(index):
        target = _target(model=f"vendor/future-{index}")
        payload = _payload(effort="medium")
        payload["model"] = target["resolved_model"]
        with request_wire_call_scope():
            candidate = prepare_wire_payload_for_send(
                target, payload, api_surface="chat.completions"
            )
            await asyncio.sleep(0)
            note_wire_send_succeeded(_capture(candidate, target, f"attempt-{index}"))
            usage = {}
            finalize_wire_response({"content": f"ok-{index}"}, usage)
            return usage, request_wire_disclosures()

    async def gather():
        return await asyncio.gather(worker(1), worker(2))

    (usage_a, disclosure_a), (usage_b, disclosure_b) = asyncio.run(gather())
    assert disclosure_a == (usage_a["request_wire"],)
    assert disclosure_b == (usage_b["request_wire"],)
    assert disclosure_a[0]["candidate_sha256"] != disclosure_b[0]["candidate_sha256"]
    assert request_wire_disclosures() == ()

    total = {}
    add_usage(total, usage_a)
    add_usage(total, usage_b)
    assert total["request_wire"] == usage_b["request_wire"]
    assert total["request_wire_history"] == [
        usage_a["request_wire"], usage_b["request_wire"],
    ]
    duplicate = {}
    merge_request_wire_usage(duplicate, total)
    merge_request_wire_usage(duplicate, total)
    assert duplicate["request_wire_history"] == total["request_wire_history"]
