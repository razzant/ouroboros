"""Exact physical-step and parser-issued receipt regressions."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros.openai_chat_custom import (
    BoundToolSchema,
    ProjectedCustomToolCatalog,
    normalize_openai_custom_tool_calls,
    project_function_tool_request_to_openai_custom,
    project_messages_for_openai_custom,
)
from ouroboros.request_wire_contract import (
    EphemeralWireAdjustment,
    PendingWireAction,
    build_request_wire_profile,
    canonical_sha256,
)
from ouroboros.request_wire_receipts import (
    REQUESTED_FUNCTION_TO_CUSTOM,
    WireAppliedAction,
    WireCandidateSpec,
    WireSemanticObservation,
    bind_wire_candidate,
    bind_wire_compatibility_receipt,
    observe_wire_semantics,
)
from ouroboros.request_wire_resolution import resolve_effort_action_set
from ouroboros.usage_accounting import PhysicalAttemptCapture


def _target():
    return {
        "provider": "openai",
        "resolved_model": "gpt-future",
        "usage_model": "openai/gpt-future",
        "base_url": "https://api.openai.example/v1",
    }


def _function_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                },
            },
        }
        for name in ("probe", "other")
    ]


def _payload(*, effort="high", choice="required"):
    return {
        "model": "gpt-future",
        "messages": [{"role": "user", "content": "x"}],
        "reasoning_effort": effort,
        "temperature": 0.2,
        "tool_choice": choice,
        "tools": _function_tools(),
    }


def _custom_payload(source, *, names=("probe", "other"), physical_choice=None):
    result = copy.deepcopy(source)
    result["tools"] = [
        {"type": "custom", "custom": {"name": name, "description": name}}
        for name in names
    ]
    if physical_choice is not None:
        result["tool_choice"] = physical_choice
    return result


def _profile(payload, *, value_fields=()):
    return build_request_wire_profile(
        _target(),
        payload,
        api_surface="chat.completions",
        evidence_scope="value" if value_fields else "capability",
        value_fields=value_fields,
    )


def _capture(candidate, *, model=None, provider_status_code=None):
    return PhysicalAttemptCapture(
        attempt_id="attempt-receipt",
        model=model or candidate.physical_model,
        provider=candidate.accepted_profile.provider,
        state="settled",
        candidate_measurement_kind="canonical_json_v1",
        candidate_raw_sha256=candidate.candidate_sha256,
        candidate_manifest_ref={
            "path": "/private/candidate.json",
            "call_id": "attempt-receipt",
            "sha256": canonical_sha256("manifest"),
        },
        provider_status_code=provider_status_code,
    )


def _function_candidate(payload=None):
    physical = payload or _payload()
    return bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=physical,
        candidate_spec=WireCandidateSpec("function", physical["reasoning_effort"], "requested_wire_form"),
        requested_effort=physical["reasoning_effort"],
        ladder_ordinal=1,
    )


def _canonical_call(call_id="call-1", name="probe", arguments='{"marker":"ok"}'):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_value_profile_retains_paths_presence_and_exact_pre_step_binding():
    source = _payload()
    profile = _profile(source, value_fields=("temperature", "reasoning_effort"))
    assert profile.value_fields == ("reasoning_effort", "temperature")
    assert profile.as_dict()["value_fields"] == ["reasoning_effort", "temperature"]

    missing = copy.deepcopy(source)
    missing.pop("temperature")
    explicit_null = copy.deepcopy(source)
    explicit_null["temperature"] = None
    assert _profile(missing, value_fields=("temperature",)) != _profile(
        explicit_null, value_fields=("temperature",)
    )

    action = PendingWireAction(profile, {
        "kind": "set_value",
        "field": "effort",
        "mode": "exact",
        "from": "high",
        "to": "medium",
        "reason_code": "provider_prescribed_value",
    })
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec("function", "medium", "provider_prescribed_value"),
        requested_effort="high",
        ladder_ordinal=2,
        applied_actions=(WireAppliedAction.pending(action),),
    )
    assert candidate.pending == (action,)

    changed_source = copy.deepcopy(source)
    changed_source["temperature"] = 0.9
    with pytest.raises(ValueError, match="exact pre-step"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=changed_source,
            candidate_spec=WireCandidateSpec("function", "medium", "provider_prescribed_value"),
            requested_effort="high",
            ladder_ordinal=2,
            applied_actions=(WireAppliedAction.pending(action),),
        )


def test_multi_action_chain_derives_every_physical_step_from_exact_profiles():
    high = _payload(effort="high")
    medium = _payload(effort="medium")
    high_action = PendingWireAction(_profile(high, value_fields=("reasoning_effort",)), {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "high", "to": "medium", "reason_code": "provider_prescribed_value",
    })
    medium_action = PendingWireAction(_profile(medium, value_fields=("reasoning_effort",)), {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "medium", "to": "low", "reason_code": "provider_prescribed_value",
    })
    applications = (
        WireAppliedAction.pending(high_action),
        WireAppliedAction.pending(medium_action),
    )
    kwargs = {
        "target": _target(),
        "api_surface": "chat.completions",
        "source_payload": high,
        "candidate_spec": WireCandidateSpec("function", "low", "provider_prescribed_value"),
        "requested_effort": "high",
        "ladder_ordinal": 3,
        "applied_actions": applications,
    }
    candidate = bind_wire_candidate(**kwargs)
    assert [item.action["from"] for item in candidate.applied_actions] == ["high", "medium"]
    assert candidate.physical_payload()["reasoning_effort"] == "low"


def test_effort_bounds_feed_exact_chains_in_order_and_cycles_fail_open():
    floor = {
        "kind": "set_value", "field": "effort", "mode": "floor",
        "from": "none", "to": "low", "reason_code": "provider_required_reasoning",
    }
    exact = {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "low", "to": "medium", "reason_code": "provider_prescribed_value",
    }
    resolved = resolve_effort_action_set("none", (exact, floor))
    assert resolved.applied_effort == "medium"
    assert [item["mode"] for item in resolved.actions] == ["floor", "exact"]

    cycle = {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "medium", "to": "low", "reason_code": "provider_prescribed_value",
    }
    conflicted = resolve_effort_action_set("low", (exact, cycle))
    assert conflicted.conflict and conflicted.applied_effort == "low"
    assert conflicted.actions == ()


def test_requested_effort_is_physical_and_nonrequested_none_requires_task_local_action():
    source = _payload(effort="high")
    with pytest.raises(ValueError, match="requested effort"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source,
            candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
            requested_effort="medium",
            ladder_ordinal=1,
        )
    with pytest.raises(ValueError, match="bound action"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source,
            candidate_spec=WireCandidateSpec("function", "none", "requested_wire_form"),
            requested_effort="high",
            ladder_ordinal=3,
        )
    adjustment = EphemeralWireAdjustment(_profile(source), {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "high", "to": "none", "reason_code": "task_local_availability_fallback",
    })
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "function", "none", "task_local_availability_fallback", task_local=True
        ),
        requested_effort="high",
        ladder_ordinal=3,
        applied_actions=(WireAppliedAction.task_local(adjustment),),
    )
    assert candidate.task_local


def test_task_local_reactive_action_only_drops_nonreasoning_fields():
    source = _payload(effort="none")
    profile = _profile(source)
    optional = PendingWireAction(profile, {
        "kind": "drop_field",
        "fields": ["temperature"],
        "reason_code": "provider_unsupported_field",
    })
    assert WireAppliedAction.reactive(optional, task_local=True).source == "task_local"

    reasoning = PendingWireAction(profile, {
        "kind": "drop_field",
        "fields": ["reasoning_effort"],
        "reason_code": "provider_unsupported_field",
    })
    with pytest.raises(ValueError, match="same-rung repair"):
        WireAppliedAction.reactive(reasoning, task_local=True)


def test_requested_custom_projection_is_first_class_and_never_committed(tmp_path):
    source = _payload()
    source["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": None, "tool_calls": [
            _canonical_call(arguments='{"marker":"prior"}')
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
    projected = project_function_tool_request_to_openai_custom(
        source["tools"], source["tool_choice"]
    )
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec("openai_chat_custom", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    assert candidate.requested_tool_projection == REQUESTED_FUNCTION_TO_CUSTOM
    assert candidate.applied_actions == () and candidate.pending == ()
    expected = copy.deepcopy(source)
    expected["messages"] = project_messages_for_openai_custom(
        source["messages"], projected.catalog
    )
    expected["tools"] = projected.catalog.wire_tools()
    expected["tool_choice"] = projected.tool_choice
    assert candidate.physical_payload() == expected
    assert candidate.custom_catalog_sha256 == projected.catalog.catalog_sha256
    assert candidate.custom_catalog_tool_names == projected.catalog.tool_names
    assert candidate.custom_catalog is not projected.catalog
    assert candidate.custom_catalog.schema_binding("probe").schema() == (
        source["tools"][0]["function"]["parameters"]
    )
    assert "custom_catalog_sha256" not in vars(candidate)
    permissive_schema = {"type": "object"}
    counterfeit = ProjectedCustomToolCatalog(
        wire_tools_json=candidate.custom_catalog.wire_tools_json,
        schemas=tuple(
            BoundToolSchema(
                name,
                '{"type":"object"}',
                canonical_sha256(permissive_schema),
            )
            for name in candidate.custom_catalog_tool_names
        ),
        catalog_sha256=candidate.custom_catalog_sha256,
        serialized_bytes=candidate.custom_catalog.serialized_bytes,
    )
    with pytest.raises(TypeError, match="custom_catalog"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source,
            candidate_spec=WireCandidateSpec(
                "openai_chat_custom", "high", "requested_wire_form"
            ),
            requested_effort="high",
            ladder_ordinal=1,
            custom_catalog=counterfeit,
        )
    source["messages"][0]["content"] = "mutated after bind"
    source["tools"][0]["function"]["description"] = "arbitrary"
    assert candidate.physical_payload() == expected

    disguised = PendingWireAction(_profile(source), {
        "kind": "replace_dialect",
        "axis": "tool",
        "from": "function",
        "to": "openai_chat_custom",
        "reason_code": "provider_rejected_tool_dialect",
    })
    with pytest.raises(ValueError, match="not a dialect action"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source,
            candidate_spec=WireCandidateSpec(
                "openai_chat_custom", "high", "requested_wire_form"
            ),
            requested_effort="high",
            ladder_ordinal=1,
            applied_actions=(WireAppliedAction.pending(disguised),),
        )

    arguments = '{"marker":"ok"}'
    normalized_calls, sidecars = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": arguments},
    }], candidate)
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"role": "assistant", "tool_calls": normalized_calls},
        normalized_usage={},
        custom_receipts=sidecars,
    )
    receipt = bind_wire_compatibility_receipt(
        candidate=candidate,
        physical_attempt=_capture(candidate),
        semantic_observation=observation,
    )
    assert wire._commit_wire_compatibility_at(tmp_path, receipt).committed
    assert not (tmp_path / "state" / wire.REQUEST_WIRE_STATE_FILE).exists()


def test_custom_text_semantics_accept_requested_projection_but_not_pending_replacement():
    source = _payload(choice="auto")
    requested = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
    )
    observation = observe_wire_semantics(
        candidate=requested,
        normalized_response={"role": "assistant", "content": "done"},
        normalized_usage={},
    )
    assert observation.semantic_kind == "chat_message"
    assert observation.custom_receipts == ()

    replacement = PendingWireAction(_profile(source), {
        "kind": "replace_dialect",
        "axis": "tool",
        "from": "function",
        "to": "openai_chat_custom",
        "reason_code": "provider_rejected_tool_dialect",
    })
    pending = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "provider_rejected_tool_dialect"
        ),
        requested_effort="high",
        ladder_ordinal=2,
        applied_actions=(WireAppliedAction.pending(replacement),),
    )
    with pytest.raises(ValueError, match="requires a custom tool call"):
        observe_wire_semantics(
            candidate=pending,
            normalized_response={"role": "assistant", "content": "done"},
            normalized_usage={},
        )


@pytest.mark.parametrize("mode", [None, "auto", "required"])
def test_allowed_tools_projection_binds_subset_mode_and_identity(mode):
    allowed_block = {"tools": [{"type": "function", "name": "probe"}]}
    if mode is not None:
        allowed_block["mode"] = mode
    choice = {
        "type": "allowed_tools",
        "allowed_tools": allowed_block,
    }
    source = _payload(choice=choice)
    projected = project_function_tool_request_to_openai_custom(source["tools"], choice)
    allowed = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec("openai_chat_custom", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    assert allowed.source_profile.tool_choice == allowed.accepted_profile.tool_choice == "allowed_tools"
    assert allowed.tool_choice_names == ("probe",)

    named_choice = {"type": "function", "function": {"name": "probe"}}
    named_source = _payload(choice=named_choice)
    named = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=named_source,
        candidate_spec=WireCandidateSpec("openai_chat_custom", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    assert named.accepted_profile.fingerprint != allowed.accepted_profile.fingerprint

    physical = allowed.physical_payload()
    assert physical["tool_choice"] == projected.tool_choice
    physical["tool_choice"] = "required" if projected.tool_choice != "required" else "auto"
    assert allowed.physical_payload()["tool_choice"] == projected.tool_choice


def test_semantics_are_parser_issued_and_bind_exact_calls_sidecars_and_candidate():
    candidate = _function_candidate()
    with pytest.raises(ValueError, match="semantic success"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={"role": "assistant", "content": "<tool_call>probe</tool_call>"},
            normalized_usage={},
        )
    response = {"role": "assistant", "tool_calls": [
        _canonical_call("call-1", "probe", '{"marker":"one"}'),
        _canonical_call("call-2", "other", '{"marker":"two"}'),
    ]}
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response=response,
        normalized_usage={},
    )
    assert [item.call_id for item in observation.tool_calls] == ["call-1", "call-2"]
    assert observation.tool_calls[0].arguments_sha256 == canonical_sha256('{"marker":"one"}')
    receipt = bind_wire_compatibility_receipt(
        candidate=candidate,
        physical_attempt=_capture(candidate),
        semantic_observation=observation,
    )
    assert receipt.observed_tool_names == ("other", "probe")

    duplicate = {"tool_calls": [_canonical_call(), _canonical_call()]}
    with pytest.raises(ValueError, match="canonical identity"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response=duplicate,
            normalized_usage={},
        )
    for malformed in (
        {"id": 1, "type": "function", "function": {"name": "probe", "arguments": "{}"}},
        {"id": "call-1", "type": "other", "function": {"name": "probe", "arguments": "{}"}},
        {"id": "call-1", "type": "function", "function": {"name": 1, "arguments": "{}"}},
    ):
        with pytest.raises(ValueError, match="canonical"):
            observe_wire_semantics(
                candidate=candidate,
                normalized_response={"tool_calls": [malformed]},
                normalized_usage={},
            )
    with pytest.raises(ValueError, match="parser-issued"):
        WireSemanticObservation(
            candidate.candidate_sha256,
            candidate.accepted_profile.fingerprint,
            canonical_sha256(response),
            "function_tool_call",
        )

    changed = _payload()
    changed["temperature"] = 0.9
    other_candidate = _function_candidate(changed)
    with pytest.raises(ValueError, match="another candidate"):
        bind_wire_compatibility_receipt(
            candidate=other_candidate,
            physical_attempt=_capture(other_candidate),
            semantic_observation=observation,
        )


def test_all_factory_authority_tokens_are_consumed_after_construction():
    source = _payload()
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
    )
    normalized_calls, custom_receipts = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }], candidate)
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"tool_calls": normalized_calls},
        normalized_usage={},
        custom_receipts=custom_receipts,
    )
    compatibility = bind_wire_compatibility_receipt(
        candidate=candidate,
        physical_attempt=_capture(candidate),
        semantic_observation=observation,
    )

    authority_objects = (
        candidate,
        observation.tool_calls[0],
        observation,
        compatibility,
        custom_receipts[0],
    )
    for authority in authority_objects:
        with pytest.raises(ValueError):
            replace(authority)


@pytest.mark.parametrize("provider_error", [
    {"kind": "provider_error", "code": 400},
    {"kind": "rate_limit", "code": 429},
    {"kind": "provider_transient", "code": 503},
])
def test_provider_body_errors_are_rejected_before_semantic_success(provider_error):
    candidate = _function_candidate(_payload(choice="auto"))
    with pytest.raises(ValueError, match="provider-error"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={"content": "apparently successful"},
            normalized_usage={"provider_error": provider_error},
        )


@pytest.mark.parametrize("response", [
    {},
    {"response_id": "resp-metadata-only"},
    {"content": "   "},
])
def test_empty_or_metadata_only_response_is_not_semantic_success(response):
    candidate = _function_candidate(_payload(choice="auto"))
    with pytest.raises(ValueError, match="non-empty response content"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response=response,
            normalized_usage={},
        )


def test_nonempty_text_tool_calls_and_unknown_success_status_remain_supported():
    text_candidate = _function_candidate(_payload(choice="auto"))
    text_observation = observe_wire_semantics(
        candidate=text_candidate,
        normalized_response={"content": "done", "tool_calls": []},
        normalized_usage={},
    )
    assert text_observation.semantic_kind == "chat_message"
    text_receipt = bind_wire_compatibility_receipt(
        candidate=text_candidate,
        physical_attempt=_capture(text_candidate, provider_status_code=None),
        semantic_observation=text_observation,
    )
    assert text_receipt.semantic_kind == "chat_message"

    tool_candidate = _function_candidate()
    tool_observation = observe_wire_semantics(
        candidate=tool_candidate,
        normalized_response={"content": "", "tool_calls": [_canonical_call()]},
        normalized_usage={},
    )
    assert tool_observation.semantic_kind == "function_tool_call"


def test_sync_and_async_normalization_reject_body_error_without_durable_evidence(
    tmp_path,
    monkeypatch,
):
    from ouroboros.llm import LLMClient

    target = {
        "provider": "openai",
        "resolved_model": "gpt-future",
        "usage_model": "openai/gpt-future",
        "base_url": "https://api.openai.example/v1",
        "supports_generation_cost": False,
    }
    body = {
        "id": "resp-body-error",
        "choices": [],
        "error": {"code": 400, "message": "invalid request"},
        "usage": {},
    }

    class Response:
        def model_dump(self):
            return copy.deepcopy(body)

    response = Response()
    endpoint = SimpleNamespace(create=lambda **_kwargs: response)
    remote_client = SimpleNamespace(
        chat=SimpleNamespace(completions=endpoint),
    )
    client = LLMClient(api_key="test")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(client, "_resolve_remote_target", lambda _model: target)
    monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote_client)
    monkeypatch.setattr(client, "_get_async_remote_client", lambda _target: remote_client)
    monkeypatch.setattr(
        client,
        "_build_remote_kwargs",
        lambda *_args, **_kwargs: {"model": "gpt-future"},
    )
    monkeypatch.setattr(client, "_normalize_payload_cache_ttl", lambda *_args: None)
    monkeypatch.setattr(
        client,
        "_create_chat_completion_with_retries",
        lambda *_args, **_kwargs: response,
    )

    async def async_response(*_args, **_kwargs):
        return response

    monkeypatch.setattr(
        client,
        "_create_chat_completion_with_retries_async",
        async_response,
    )

    messages = [{"role": "user", "content": "hello"}]
    sync_message, sync_usage = client.chat(messages, "openai/gpt-future")
    async_message, async_usage = asyncio.run(
        client.chat_async(messages, "openai/gpt-future")
    )
    assert sync_usage["provider_error"]["kind"] == "provider_error"
    assert async_usage["provider_error"] == sync_usage["provider_error"]
    assert sync_usage["ledger_attempt_ids"] == []
    assert async_usage["ledger_attempt_ids"] == []

    candidate = _function_candidate(_payload(choice="auto"))
    for message, usage in (
        (sync_message, sync_usage),
        (async_message, async_usage),
    ):
        with pytest.raises(ValueError, match="provider-error"):
            observe_wire_semantics(
                candidate=candidate,
                normalized_response=message,
                normalized_usage=usage,
            )
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_allowed_tools_required_rejects_plain_text_and_requested_projection_composes_durable():
    choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "required",
            "tools": [{"type": "function", "name": "probe"}],
        },
    }
    source = _payload(choice=choice)
    after_drop = copy.deepcopy(source)
    after_drop.pop("temperature")
    drop = PendingWireAction(_profile(source), {
        "kind": "drop_field",
        "fields": ["temperature"],
        "reason_code": "provider_unsupported_field",
    })
    projected = project_function_tool_request_to_openai_custom(
        after_drop["tools"], choice
    )
    durable = WireAppliedAction(drop.profile, drop.action, "durable")
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
        applied_actions=(durable,),
    )
    assert candidate.requested_tool_projection == REQUESTED_FUNCTION_TO_CUSTOM
    assert candidate.pending == ()
    assert candidate.disclosed_actions()[0]["source"] == "durable"
    custom_pre_step = copy.deepcopy(source)
    custom_pre_step["tools"] = projected.catalog.wire_tools()
    custom_pre_step["tool_choice"] = projected.tool_choice
    custom_profile = build_request_wire_profile(
        _target(),
        custom_pre_step,
        api_surface="chat.completions",
        tool_choice_override=choice,
    )
    custom_drop = PendingWireAction(
        custom_profile,
        {
        "kind": "drop_field",
        "fields": ["temperature"],
        "reason_code": "provider_unsupported_field",
        },
    )
    custom_ordered = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
        applied_actions=(
            WireAppliedAction(custom_drop.profile, custom_drop.action, "durable"),
        ),
    )
    assert "temperature" not in custom_ordered.physical_payload()
    assert custom_ordered.physical_payload()["tools"] == projected.catalog.wire_tools()
    with pytest.raises(ValueError, match="semantic success|required a tool call"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={"role": "assistant", "content": "no call"},
            normalized_usage={},
        )


def test_custom_semantic_sidecar_must_match_call_argument_digest():
    source = _payload()
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec("openai_chat_custom", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    _, sidecars = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"different"}'},
    }], candidate)
    with pytest.raises(ValueError, match="sidecar differs"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={"tool_calls": [_canonical_call()]},
            normalized_usage={},
            custom_receipts=sidecars,
        )


def test_schema_invalid_custom_call_proves_wire_but_never_execution():
    source = _payload()
    source["tools"][0]["function"]["parameters"]["properties"]["marker"] = {
        "const": "ok"
    }
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
    )
    calls, receipts = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"wrong"}'},
    }], candidate)
    assert receipts[0].proves_wire_acceptance
    assert not receipts[0].allows_execution

    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"content": "", "tool_calls": calls},
        normalized_usage={},
        custom_receipts=receipts,
    )
    assert observation.custom_receipts[0].error_code == "schema_validation_failed"
    assert not observation.custom_receipts[0].allows_execution


def test_model_identity_maps_resolved_payload_to_accounting_capture_exactly():
    candidate = _function_candidate()
    assert candidate.custom_catalog is None
    assert candidate.accepted_profile.model == "gpt-future"
    assert candidate.physical_payload()["model"] == "gpt-future"
    assert candidate.physical_model == "openai/gpt-future"

    mismatched = _payload()
    mismatched["model"] = "model-b"
    with pytest.raises(ValueError, match="payload model differs"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=mismatched,
            candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
            requested_effort="high",
            ladder_ordinal=1,
        )

    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"tool_calls": [_canonical_call()]},
        normalized_usage={},
    )
    with pytest.raises(ValueError, match="another route"):
        bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=_capture(candidate, model="gpt-future"),
            semantic_observation=observation,
        )
    fallback_target = _target()
    fallback_target.pop("usage_model")
    fallback = bind_wire_candidate(
        target=fallback_target,
        api_surface="chat.completions",
        source_payload=_payload(),
        candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    assert fallback.physical_model == "gpt-future"


def test_applied_action_snapshot_is_recursively_immutable_and_commits_bound_fact(tmp_path):
    source = _payload()
    raw_action = {
        "kind": "drop_field",
        "fields": ["temperature"],
        "reason_code": "provider_unsupported_field",
    }
    pending = PendingWireAction(_profile(source), raw_action)
    applied = WireAppliedAction.pending(pending)
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source,
        candidate_spec=WireCandidateSpec(
            "function", "high", "provider_unsupported_field"
        ),
        requested_effort="high",
        ladder_ordinal=2,
        applied_actions=(applied,),
    )
    raw_action["fields"][0] = "top_p"
    with pytest.raises(TypeError):
        applied.action["fields"][0] = "top_p"
    with pytest.raises(TypeError):
        candidate.applied_actions[0].action["fields"] = ("top_p",)
    assert "temperature" not in candidate.physical_payload()
    assert "top_p" not in candidate.physical_payload()

    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={"tool_calls": [_canonical_call()]},
        normalized_usage={},
    )
    receipt = bind_wire_compatibility_receipt(
        candidate=candidate,
        physical_attempt=_capture(candidate),
        semantic_observation=observation,
    )
    assert wire._commit_wire_compatibility_at(tmp_path, receipt).committed
    committed = wire._read_wire_actions_at(tmp_path, pending.profile)
    assert committed[0]["fields"] == ["temperature"]
