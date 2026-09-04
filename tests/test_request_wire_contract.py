"""Foundation contracts for route-scoped provider wire adaptation."""

from __future__ import annotations

import json
import multiprocessing
from datetime import timedelta

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros.deadline_utils import utc_now
from ouroboros.openai_chat_custom import (
    normalize_openai_custom_tool_calls,
    project_function_tool_request_to_openai_custom,
)
from ouroboros.request_wire_contract import (
    REQUEST_WIRE_STATE_FILE,
    EphemeralWireAdjustment,
    PendingWireAction,
    StoredWireAction,
    apply_effort_action,
    build_request_wire_profile,
    canonical_sha256,
    normalize_endpoint,
    payload_effort,
    validate_durable_wire_action_for_profile,
    validate_wire_action,
    validate_wire_action_for_profile,
)
from ouroboros.request_wire_receipts import (
    CustomArgumentValidationReceipt,
    WireAppliedAction,
    WireCandidateSpec,
    WireUsageDisclosure,
    bind_wire_candidate,
    bind_wire_compatibility_receipt,
    direct_openai_tool_candidate_ladder,
    observe_wire_semantics,
    wire_semantic_kind_allowed,
)
from ouroboros.request_wire_resolution import resolve_wire_actions
from ouroboros.usage_accounting import PhysicalAttemptCapture


def _target(provider="openai", model="gpt-future", base_url="https://API.Example/v1/"):
    return {
        "provider": provider,
        "resolved_model": model,
        "usage_model": f"{provider}/{model}",
        "base_url": base_url,
        "api_key": "must-not-enter-profile",
    }


def _payload(*, effort="high", tools=True, tool_choice="required"):
    payload = {
        "model": "gpt-future",
        "messages": [{"role": "user", "content": "x"}],
        "reasoning_effort": effort,
        "tool_choice": tool_choice,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = [{
            "type": "function",
            "function": {
                "name": "probe",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                },
            },
        }]
    return payload


def _profile(**kwargs):
    target = kwargs.pop("target", _target())
    payload = kwargs.pop("payload", None)
    if payload is None:
        payload = _payload()
        payload["model"] = target["resolved_model"]
    return build_request_wire_profile(
        target,
        payload,
        api_surface=kwargs.pop("api_surface", "chat.completions"),
        **kwargs,
    )


def _pending(profile=None, action=None):
    return PendingWireAction(profile or _profile(), action or {
        "kind": "drop_field",
        "fields": ["temperature"],
        "reason_code": "provider_unsupported_field",
    })


def _capture(
    candidate,
    *,
    state="settled",
    candidate_sha256=None,
    provider_status_code=None,
):
    attempt_id = "attempt-1"
    return PhysicalAttemptCapture(
        attempt_id=attempt_id,
        model=candidate.physical_model,
        provider=candidate.accepted_profile.provider,
        state=state,
        candidate_measurement_kind="canonical_json_v1",
        candidate_raw_sha256=candidate_sha256 or candidate.candidate_sha256,
        candidate_manifest_ref={
            "path": "/private/physical-candidate.json",
            "call_id": attempt_id,
            "sha256": canonical_sha256("physical-manifest"),
        },
        provider_status_code=provider_status_code,
    )


def _candidate(profile=None, pending=()):
    accepted = profile or _profile()
    target = _target(provider=accepted.provider, model=accepted.model)
    source_payload = _payload()
    source_payload["model"] = accepted.model
    source_payload["tool_choice"] = accepted.tool_choice
    candidate = bind_wire_candidate(
        target=target,
        api_surface=accepted.api_surface,
        source_payload=source_payload,
        candidate_spec=WireCandidateSpec(
            accepted.tool_dialect, "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
        applied_actions=tuple(WireAppliedAction.pending(item) for item in pending),
    )
    assert candidate.source_profile.fingerprint == accepted.fingerprint
    return candidate


def _custom_candidate():
    payload = _payload()
    return bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=payload,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form"
        ),
        requested_effort="high",
        ladder_ordinal=1,
    )


def _receipt(profile=None, pending=()):
    candidate = _custom_candidate() if (
        profile is not None and profile.tool_dialect == "openai_chat_custom"
    ) else _candidate(profile, pending)
    custom_receipts = ()
    normalized_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "probe", "arguments": "{}"},
        }],
    }
    if candidate.accepted_profile.tool_dialect == "openai_chat_custom":
        normalized_calls, custom_receipts = normalize_openai_custom_tool_calls([{
            "id": "call-1",
            "type": "custom",
            "custom": {"name": "probe", "input": '{"marker":"ok"}'},
        }], candidate)
        normalized_response["tool_calls"] = normalized_calls
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response=normalized_response,
        normalized_usage={},
        custom_receipts=custom_receipts,
    )
    return bind_wire_compatibility_receipt(
        candidate=candidate,
        physical_attempt=_capture(candidate),
        semantic_observation=observation,
    )


def test_endpoint_profile_is_secret_free_exact_and_ipv6_safe():
    first = _profile()
    changed_key = _target()
    changed_key["api_key"] = "different-secret"
    assert first == _profile(target=changed_key)
    assert "secret" not in json.dumps(first.as_dict())

    safe = normalize_endpoint(
        "https://user:pass@[::1]:443/v1/?api_key=SECRET&api-version=2025#fragment"
    )
    assert safe == "https://[::1]:443/v1?api-version=2025"
    assert "SECRET" not in safe
    assert normalize_endpoint("api.example/v1?api_key=SECRET") == ""
    assert normalize_endpoint("https://[::1]:443/v1") != normalize_endpoint("https://[::1:443]/v1")

    old = _profile(target=_target(base_url="https://api.example/v1?api-version=2024"))
    new = _profile(target=_target(base_url="https://api.example/v1?api-version=2025"))
    assert old.endpoint_sha256 != new.endpoint_sha256


def test_profile_rejects_incomplete_identity_but_accepts_future_model():
    with pytest.raises(ValueError, match="incomplete"):
        build_request_wire_profile({}, {}, api_surface="")
    assert _profile(target=_target(model="future-model-2099")).model == "future-model-2099"
    with pytest.raises(ValueError, match="payload model differs"):
        _profile(target=_target(model="model-a"), payload=_payload())
    default_endpoint = _profile(target=_target(base_url=""))
    assert len(default_endpoint.endpoint_sha256) == 64


def test_profile_isolates_route_surface_model_and_request_dialect_without_optional_noise():
    base = _profile()
    without_temperature = _payload()
    without_temperature.pop("temperature")
    with_top_p = _payload()
    with_top_p["top_p"] = 0.9
    assert base.fingerprint == _profile(payload=without_temperature).fingerprint
    assert base.fingerprint == _profile(payload=with_top_p).fingerprint

    variants = [
        _profile(target=_target(provider="openrouter")),
        _profile(target=_target(base_url="https://other.example/v1")),
        _profile(api_surface="responses"),
        _profile(target=_target(model="gpt-other")),
        _profile(payload=_payload(tools=False)),
        _profile(tool_dialect_override="openai_chat_custom"),
        _profile(payload=_payload(tool_choice="auto")),
        _profile(function_strictness_override="plain"),
    ]
    assert len({base.fingerprint, *(item.fingerprint for item in variants)}) == len(variants) + 1

    named_probe = _profile(payload=_payload(tool_choice={
        "type": "function", "function": {"name": "probe"},
    }))
    named_other = _profile(payload=_payload(tool_choice={
        "type": "function", "function": {"name": "other"},
    }))
    assert named_probe.tool_choice == named_other.tool_choice == "named"
    assert named_probe.fingerprint != named_other.fingerprint


def test_profile_isolates_provider_routing_reasoning_options_and_semantic_headers():
    left = _payload()
    left["extra_body"] = {
        "provider": {"order": ["anthropic"], "allow_fallbacks": False},
        "reasoning": {"effort": "high", "exclude": False},
    }
    left.pop("reasoning_effort")
    right = json.loads(json.dumps(left))
    right["extra_body"]["provider"] = {"zdr": True, "max_price": {"prompt": 2}}
    excluded = json.loads(json.dumps(left))
    excluded["extra_body"]["reasoning"]["exclude"] = True
    a = _profile(target=_target(provider="openrouter"), payload=left)
    b = _profile(target=_target(provider="openrouter"), payload=right)
    c = _profile(target=_target(provider="openrouter"), payload=excluded)
    assert len({a.fingerprint, b.fingerprint, c.fingerprint}) == 3

    header_a = _target()
    header_a["contract_headers"] = {"anthropic-version": "1", "x-goog-api-key": "secret-a"}
    header_b = _target()
    header_b["contract_headers"] = {"anthropic-version": "1", "x-goog-api-key": "secret-b"}
    header_c = _target()
    header_c["contract_headers"] = {"anthropic-version": "2", "x-goog-api-key": "secret-a"}
    assert _profile(target=header_a).fingerprint == _profile(target=header_b).fingerprint
    assert _profile(target=header_a).fingerprint != _profile(target=header_c).fingerprint


def test_value_scope_is_exact_while_capability_scope_generalizes():
    low = _payload()
    high = _payload()
    low["temperature"] = 0.2
    high["temperature"] = 7.0
    assert _profile(payload=low).fingerprint == _profile(payload=high).fingerprint
    low_value = _profile(payload=low, evidence_scope="value", value_fields=("temperature",))
    high_value = _profile(payload=high, evidence_scope="value", value_fields=("temperature",))
    assert low_value.fingerprint != high_value.fingerprint
    with pytest.raises(ValueError, match="value field"):
        _profile(evidence_scope="value")


def test_effort_actions_are_source_bounded_and_aggregate_order_independent():
    ceiling = validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "ceiling",
        "from": "high", "to": "medium", "reason_code": "provider_prescribed_value",
    })
    assert apply_effort_action("max", ceiling) == "medium"
    assert apply_effort_action("high", ceiling) == "medium"
    assert apply_effort_action("low", ceiling) == "low"

    floor = validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "floor",
        "from": "none", "to": "low", "reason_code": "provider_required_reasoning",
    })
    assert apply_effort_action("none", floor) == "low"
    assert apply_effort_action("high", floor) == "high"

    conflict_floor = StoredWireAction(validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "floor",
        "from": "none", "to": "medium", "reason_code": "provider_required_reasoning",
    }), "2026-08-19T00:00:00+00:00")
    conflict_ceiling = StoredWireAction(validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "ceiling",
        "from": "high", "to": "low", "reason_code": "provider_prescribed_value",
    }), "2026-08-19T00:00:01+00:00")
    forward = resolve_wire_actions(
        _profile(), requested_effort="none", records=(conflict_floor, conflict_ceiling)
    )
    reverse = resolve_wire_actions(
        _profile(), requested_effort="none", records=(conflict_ceiling, conflict_floor)
    )
    assert forward == reverse
    assert forward.effort_conflict and forward.applied_effort == "none"

    impossible_floor = StoredWireAction(validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "floor",
        "from": "none", "to": "high", "reason_code": "provider_required_reasoning",
    }), "2026-08-19T00:00:00+00:00")
    impossible_ceiling = StoredWireAction(validate_wire_action({
        "kind": "set_value", "field": "effort", "mode": "ceiling",
        "from": "max", "to": "low", "reason_code": "provider_prescribed_value",
    }), "2026-08-19T00:00:01+00:00")
    conflict = resolve_wire_actions(
        _profile(), requested_effort="medium", records=(impossible_floor, impossible_ceiling)
    )
    assert conflict.effort_conflict and conflict.applied_effort == "medium"


def test_action_vocabulary_is_closed_and_profile_aware():
    assert not validate_wire_action({"kind": "switch_model", "to": "other"})
    assert not validate_wire_action({"kind": "switch_api", "to": "responses"})
    assert not validate_wire_action({
        "kind": "drop_field", "fields": ["temperature", "model"],
    })
    assert not validate_wire_action({
        "kind": "drop_field", "fields": ["temperature"], "reason_code": "Bearer secret",
    })
    replace = {
        "kind": "replace_dialect", "axis": "tool",
        "from": "function", "to": "openai_chat_custom",
        "reason_code": "provider_rejected_tool_dialect",
    }
    assert validate_wire_action_for_profile(_profile(), replace)["to"] == "openai_chat_custom"
    assert not validate_wire_action_for_profile(_profile(payload=_payload(tools=False)), replace)

    anthropic_payload = {
        "model": "claude-future",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    anthropic = _profile(
        target=_target(provider="anthropic", model="claude-future"),
        payload=anthropic_payload,
        api_surface="messages",
    )
    linked = validate_wire_action_for_profile(anthropic, {
        "kind": "drop_field", "fields": ["thinking"],
        "reason_code": "provider_unsupported_field",
    })
    assert linked["fields"] == ["output_config", "thinking"]


def test_explicit_none_is_a_distinct_ephemeral_type_and_never_pending():
    explicit_none = {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "medium", "to": "none",
        "reason_code": "task_local_availability_fallback",
    }
    profile = _profile()
    adjustment = EphemeralWireAdjustment(profile, explicit_none)
    assert adjustment.action["to"] == "none"
    with pytest.raises(ValueError, match="invalid request-wire action"):
        PendingWireAction(profile, explicit_none)
    assert not validate_durable_wire_action_for_profile(profile, explicit_none)


def test_candidate_ladder_preserves_owner_order_and_caller_rail():
    two = direct_openai_tool_candidate_ladder("medium", remaining_physical_attempts=2)
    assert [(item.tool_dialect, item.effort) for item in two] == [
        ("openai_chat_custom", "medium"),
        ("function", "medium"),
    ]
    three = direct_openai_tool_candidate_ladder("medium", remaining_physical_attempts=3)
    assert three[-1].effort == "none" and three[-1].task_local
    assert direct_openai_tool_candidate_ladder("medium", remaining_physical_attempts=0) == ()
    owner_none = direct_openai_tool_candidate_ladder("none", remaining_physical_attempts=2)
    assert len(owner_none) == 1 and not owner_none[0].task_local
    assert owner_none[0].reason_code == "requested_wire_form"
    assert two[0].reason_code == "requested_wire_form"


def test_terminal_receipt_enforces_semantic_success_and_physical_authority():
    candidate = _custom_candidate()
    observation = _receipt(candidate.accepted_profile).semantic_observation
    with pytest.raises(ValueError, match="non-successful provider settlement"):
        bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=_capture(candidate, provider_status_code=404),
            semantic_observation=observation,
        )
    with pytest.raises(ValueError, match="settled physical attempt"):
        bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=_capture(candidate, state="released"),
            semantic_observation=observation,
        )
    with pytest.raises(ValueError, match="semantic success"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={"response": "ok"},
            normalized_usage={},
        )
    with pytest.raises(ValueError, match="candidate digest"):
        bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=_capture(
                candidate, candidate_sha256=canonical_sha256("wrong")
            ),
            semantic_observation=observation,
        )
    with pytest.raises(ValueError, match="sidecar"):
        observe_wire_semantics(
            candidate=candidate,
            normalized_response={
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }],
            },
            normalized_usage={},
        )

    required = _profile()
    required_candidate = _candidate(required)
    with pytest.raises(ValueError, match="semantic success"):
        observe_wire_semantics(
            candidate=required_candidate,
            normalized_response={"response": "text"},
            normalized_usage={},
        )

    foreign_pending = _pending(_profile(target=_target(provider="openrouter")))
    with pytest.raises(ValueError, match="candidate route"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=_payload(),
            candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
            requested_effort="high",
            ladder_ordinal=1,
            applied_actions=(WireAppliedAction.pending(foreign_pending),),
        )


def test_candidate_manifest_binds_applied_transform_and_reasoning_shape():
    function = _profile()
    replace = PendingWireAction(function, {
        "kind": "replace_dialect",
        "axis": "tool",
        "from": "function",
        "to": "openai_chat_custom",
        "reason_code": "provider_rejected_tool_dialect",
    })
    source_payload = _payload()
    with pytest.raises(ValueError, match="candidate spec dialect"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source_payload,
            candidate_spec=WireCandidateSpec(
                "function", "high", "provider_rejected_tool_dialect"
            ),
            requested_effort="high",
            ladder_ordinal=2,
            applied_actions=(WireAppliedAction.pending(replace),),
        )
    repaired = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source_payload,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "provider_rejected_tool_dialect"
        ),
        requested_effort="high",
        ladder_ordinal=2,
        applied_actions=(WireAppliedAction.pending(replace),),
    )
    assert repaired.pending[0].action["to"] == "openai_chat_custom"
    assert repaired.physical_payload()["tools"][0]["type"] == "custom"

    nested = _payload()
    nested.pop("reasoning_effort")
    nested["extra_body"] = {"reasoning": {"effort": "high", "exclude": False}}
    owned = bind_wire_candidate(
        target=_target(provider="openrouter"),
        api_surface="chat.completions",
        source_payload=nested,
        candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    decoded = owned.physical_payload()
    decoded["extra_body"]["reasoning"]["exclude"] = True
    assert owned.physical_payload()["extra_body"]["reasoning"]["exclude"] is False

    drop = _pending(function)
    dropped = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source_payload,
        candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
        applied_actions=(WireAppliedAction.pending(drop),),
    )
    assert "temperature" not in dropped.physical_payload()


def test_candidate_binding_composes_dialect_drop_and_exact_effort_chain():
    function_mid = _payload()
    projected = project_function_tool_request_to_openai_custom(
        function_mid["tools"], function_mid["tool_choice"]
    )
    custom_source = json.loads(json.dumps(function_mid))
    custom_source["tools"] = projected.catalog.wire_tools()
    custom_source["tool_choice"] = projected.tool_choice
    custom_profile = _profile(
        payload=custom_source, tool_dialect_override="openai_chat_custom"
    )
    dialect = PendingWireAction(custom_profile, {
        "kind": "replace_dialect",
        "axis": "tool",
        "from": "openai_chat_custom",
        "to": "function",
        "reason_code": "provider_rejected_tool_dialect",
    })
    drop = _pending(_profile(payload=function_mid))
    composed = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=function_mid,
        candidate_spec=WireCandidateSpec(
            "function", "high", "provider_rejected_tool_dialect"
        ),
        requested_effort="high",
        ladder_ordinal=2,
        applied_actions=(
            WireAppliedAction.pending(dialect),
            WireAppliedAction.pending(drop),
        ),
    )
    assert [item.action["kind"] for item in composed.pending] == [
        "replace_dialect", "drop_field",
    ]
    assert composed.physical_payload()["tools"] == function_mid["tools"]
    assert "temperature" not in composed.physical_payload()

    high_payload = _payload(effort="high")
    medium_payload = _payload(effort="medium")
    high_to_medium = PendingWireAction(_profile(payload=high_payload), {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "high", "to": "medium", "reason_code": "provider_prescribed_value",
    })
    medium_to_low = PendingWireAction(_profile(payload=medium_payload), {
        "kind": "set_value", "field": "effort", "mode": "exact",
        "from": "medium", "to": "low", "reason_code": "provider_prescribed_value",
    })
    effort_chain = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=high_payload,
        candidate_spec=WireCandidateSpec("function", "low", "provider_prescribed_value"),
        requested_effort="high",
        ladder_ordinal=3,
        applied_actions=(
            WireAppliedAction.pending(high_to_medium),
            WireAppliedAction.pending(medium_to_low),
        ),
    )
    assert len(effort_chain.pending) == 2


def test_receipt_is_parser_issued_and_binds_exact_custom_schema():
    custom = _custom_candidate()
    with pytest.raises(ValueError, match="parser-issued"):
        CustomArgumentValidationReceipt(
            candidate_sha256=custom.candidate_sha256,
            catalog_sha256=custom.custom_catalog_sha256,
            tool_call_id="call-1",
            tool_name="probe",
            schema_sha256=canonical_sha256("schema"),
            arguments_sha256=canonical_sha256("{}"),
            decoded_object=True,
            valid=True,
        )

    calls, receipts = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }], custom)
    wrong_schema = receipts[0]
    object.__setattr__(wrong_schema, "schema_sha256", canonical_sha256("wrong-schema"))
    with pytest.raises(ValueError, match="sidecar"):
        observe_wire_semantics(
            candidate=custom,
            normalized_response={"tool_calls": calls},
            normalized_usage={},
            custom_receipts=(wrong_schema,),
        )


def test_receipt_binds_exact_named_tool_choice():

    named_payload = _payload(tool_choice={
        "type": "function", "function": {"name": "probe"},
    })
    named = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=named_payload,
        candidate_spec=WireCandidateSpec("function", "high", "requested_wire_form"),
        requested_effort="high",
        ladder_ordinal=1,
    )
    assert named.tool_choice_names == ("probe",)
    with pytest.raises(ValueError, match="outside the exact tool-choice"):
        observe_wire_semantics(
            candidate=named,
            normalized_response={"tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "other", "arguments": "{}"},
            }]},
            normalized_usage={},
        )
    observation = observe_wire_semantics(
        candidate=named,
        normalized_response={"tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "probe", "arguments": "{}"},
        }]},
        normalized_usage={},
    )
    accepted = bind_wire_compatibility_receipt(
        candidate=named,
        physical_attempt=_capture(named),
        semantic_observation=observation,
    )
    assert accepted.observed_tool_names == ("probe",)


def test_task_local_none_and_disclosure_are_candidate_and_attempt_bound():
    profile = _profile()
    adjustment = EphemeralWireAdjustment(profile, {
        "kind": "set_value",
        "field": "effort",
        "mode": "exact",
        "from": "high",
        "to": "none",
        "reason_code": "task_local_availability_fallback",
    })
    source_payload = _payload()
    candidate = bind_wire_candidate(
        target=_target(),
        api_surface="chat.completions",
        source_payload=source_payload,
        candidate_spec=WireCandidateSpec(
            "function", "none", "task_local_availability_fallback", task_local=True
        ),
        requested_effort="high",
        ladder_ordinal=3,
        applied_actions=(WireAppliedAction.task_local(adjustment),),
    )
    capture = _capture(candidate)
    disclosure = WireUsageDisclosure.from_candidate(candidate, capture)
    payload = disclosure.as_dict()
    assert payload["requested_effort"] == "high"
    assert payload["applied_effort"] == "none"
    assert payload["task_local"] is True
    assert payload["attempt_id"] == capture.attempt_id
    assert payload["candidate_sha256"] == candidate.candidate_sha256
    assert payload["applied_actions"] == [{
        "source": "task_local",
        "profile_fingerprint": profile.fingerprint,
        "action": wire._wire_action_dict(adjustment.action),
    }]
    assert payload["ladder_ordinal"] == 3
    with pytest.raises(ValueError, match="task-local status"):
        bind_wire_candidate(
            target=_target(),
            api_surface="chat.completions",
            source_payload=source_payload,
            candidate_spec=WireCandidateSpec("function", "none", "requested_wire_form"),
            requested_effort="high",
            ladder_ordinal=1,
            applied_actions=(WireAppliedAction.task_local(adjustment),),
        )


def test_payload_effort_and_semantic_matrix_cover_all_current_carriers():
    assert payload_effort({"reasoning_effort": "medium"}) == "medium"
    assert payload_effort({
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}
    }) == "high"
    assert payload_effort({"thinking": {"type": "disabled"}}) == "none"
    assert payload_effort({
        "extra_body": {"reasoning": {"effort": "low", "exclude": False}}
    }) == "low"

    required = _profile()
    auto = _profile(payload=_payload(tool_choice="auto"))
    none = _profile(payload=_payload(tool_choice="none"))
    assert wire_semantic_kind_allowed(required, "function_tool_call")
    assert not wire_semantic_kind_allowed(required, "chat_message")
    assert wire_semantic_kind_allowed(auto, "chat_message")
    assert wire_semantic_kind_allowed(auto, "function_tool_call")
    assert wire_semantic_kind_allowed(none, "chat_message")
    assert not wire_semantic_kind_allowed(none, "function_tool_call")

    def _custom_profile(choice):
        source = _payload(tool_choice=choice)
        projected = project_function_tool_request_to_openai_custom(
            source["tools"], choice
        )
        source["tools"] = projected.catalog.wire_tools()
        source["tool_choice"] = projected.tool_choice
        return _profile(payload=source)

    custom_required = _custom_profile("required")
    custom_auto = _custom_profile("auto")
    custom_none = _custom_profile("none")
    assert wire_semantic_kind_allowed(
        custom_required, "custom_tool_call_json_object"
    )
    assert not wire_semantic_kind_allowed(custom_required, "chat_message")
    assert wire_semantic_kind_allowed(custom_auto, "chat_message")
    assert wire_semantic_kind_allowed(
        custom_auto, "custom_tool_call_json_object"
    )
    assert wire_semantic_kind_allowed(custom_none, "chat_message")
    assert not wire_semantic_kind_allowed(
        custom_none, "custom_tool_call_json_object"
    )

    anthropic_payload = {
        "model": "claude-future",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "tools": [{"name": "probe", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "probe"},
    }
    anthropic = _profile(
        target=_target(provider="anthropic", model="claude-future"),
        payload=anthropic_payload,
        api_surface="messages",
    )
    assert wire_semantic_kind_allowed(anthropic, "anthropic_tool_use")
    assert not wire_semantic_kind_allowed(anthropic, "anthropic_message")


def test_store_commits_only_typed_receipt_and_uses_canonical_root(tmp_path, monkeypatch):
    profile = _profile()
    pending = (_pending(profile),)
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    result = wire.commit_wire_compatibility(_receipt(profile, pending))
    assert result.committed
    assert wire.read_wire_actions(profile)[0]["kind"] == "drop_field"


def test_store_ttl_future_time_and_malformed_rows_fail_open(tmp_path):
    profile = _profile()
    pending = (_pending(profile),)
    assert wire._commit_wire_compatibility_at(tmp_path, _receipt(profile, pending)).committed
    path = tmp_path / "state" / REQUEST_WIRE_STATE_FILE
    data = json.loads(path.read_text())
    record = data["profiles"][profile.fingerprint]["records"][0]
    record["observed_at"] = (utc_now() - timedelta(days=30)).isoformat()
    data["profiles"][profile.fingerprint]["records"].append({
        "action": {"kind": "set_value", "field": "effort", "mode": "exact",
                   "from": "low", "to": "garbage"},
        "observed_at": utc_now().isoformat(),
    })
    path.write_text(json.dumps(data))
    assert wire._read_wire_actions_at(tmp_path, profile) == ()

    record["observed_at"] = (utc_now() + timedelta(days=3650)).isoformat()
    path.write_text(json.dumps(data))
    assert wire._read_wire_actions_at(tmp_path, profile) == ()


def test_store_preserves_malformed_future_and_misbound_state(tmp_path):
    profile = _profile()
    pending = (_pending(profile),)
    path = tmp_path / "state" / REQUEST_WIRE_STATE_FILE
    path.parent.mkdir(parents=True)

    malformed = "{malformed-state"
    path.write_text(malformed)
    result = wire._commit_wire_compatibility_at(tmp_path, _receipt(profile, pending))
    assert not result.committed and result.error_code == "incompatible_state"
    assert path.read_text() == malformed

    future = {"schema_version": 2, "profiles": {"future": {"kept": True}}}
    path.write_text(json.dumps(future))
    result = wire._commit_wire_compatibility_at(tmp_path, _receipt(profile, pending))
    assert not result.committed and result.error_code == "incompatible_state"
    assert json.loads(path.read_text()) == future

    stale_action = {
        "kind": "set_value", "field": "effort", "mode": "ceiling",
        "from": "high", "to": "low", "reason_code": "provider_prescribed_value",
    }
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {profile.fingerprint: {
            "profile": {"corrupt": True},
            "records": [{"action": stale_action, "observed_at": utc_now().isoformat()}],
        }},
    }))
    assert wire._read_wire_actions_at(tmp_path, profile) == ()
    assert wire._commit_wire_compatibility_at(tmp_path, _receipt(profile, pending)).committed
    actions = wire._read_wire_actions_at(tmp_path, profile)
    assert len(actions) == 1 and actions[0]["kind"] == "drop_field"


@pytest.mark.parametrize("schema_version", ["1", True, 1.0])
def test_store_rejects_coerced_schema_versions(tmp_path, schema_version):
    profile = _profile()
    path = tmp_path / "state" / REQUEST_WIRE_STATE_FILE
    path.parent.mkdir(parents=True)
    seeded = {
        "schema_version": schema_version,
        "profiles": {profile.fingerprint: {
            "profile": profile.as_dict(),
            "records": [{
                "action": dict(_pending(profile).action),
                "observed_at": utc_now().isoformat(),
            }],
        }},
    }
    path.write_text(json.dumps(seeded))
    before = path.read_bytes()
    assert wire._read_wire_actions_at(tmp_path, profile) == ()
    result = wire._commit_wire_compatibility_at(
        tmp_path, _receipt(profile, (_pending(profile),))
    )
    assert not result.committed and result.error_code == "incompatible_state"
    assert path.read_bytes() == before


def test_store_rejects_seeded_durable_none_and_preserves_existing_empty_state(tmp_path):
    profile = _profile()
    path = tmp_path / "state" / REQUEST_WIRE_STATE_FILE
    path.parent.mkdir(parents=True)
    explicit_none = {
        "kind": "set_value",
        "field": "effort",
        "mode": "ceiling",
        "from": "high",
        "to": "none",
        "reason_code": "provider_prescribed_value",
    }
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {profile.fingerprint: {
            "profile": profile.as_dict(),
            "records": [{"action": explicit_none, "observed_at": utc_now().isoformat()}],
        }},
    }))
    assert wire._read_wire_actions_at(tmp_path, profile) == ()

    path.write_text("{}")
    result = wire._commit_wire_compatibility_at(
        tmp_path, _receipt(profile, (_pending(profile),))
    )
    assert not result.committed and result.error_code == "incompatible_state"
    assert path.read_text() == "{}"


def _process_writer(root: str, model: str) -> None:
    profile = _profile(target=_target(model=model))
    pending = (_pending(profile),)
    result = wire._commit_wire_compatibility_at(root, _receipt(profile, pending))
    if not result.committed:
        raise RuntimeError(result.error_code)


@pytest.mark.serial
def test_store_cross_process_writers_do_not_lose_updates(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_process_writer, args=(str(tmp_path), f"model-{index}"))
        for index in range(4)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    for index in range(4):
        profile = _profile(target=_target(model=f"model-{index}"))
        assert wire._read_wire_actions_at(tmp_path, profile)[0]["kind"] == "drop_field"


def test_disclosure_survives_a_failed_post_response_settle_but_learning_does_not():
    """CyberGym r8 (2026-09-04): a response whose ledger settle failed under a
    lock convoy was captured ``unresolved``; the disclosure then raised, the
    served call carried no ``applied_effort``, and the benchmark executor
    refused the whole task. The wire facts do not depend on the ledger."""
    candidate = _candidate()
    unresolved = _capture(candidate, state="unresolved")

    disclosure = WireUsageDisclosure.from_candidate(candidate, unresolved)
    assert disclosure.applied_effort == "high"
    assert disclosure.attempt_id == unresolved.attempt_id
    assert disclosure.candidate_sha256 == candidate.candidate_sha256

    # A send that never produced a response still cannot be disclosed.
    for state in ("reserved", "dispatched", "released"):
        with pytest.raises(ValueError, match="settled physical attempt"):
            WireUsageDisclosure.from_candidate(candidate, _capture(candidate, state=state))

    # Compatibility-profile learning keeps requiring settled accounting.
    observation = observe_wire_semantics(
        candidate=candidate,
        normalized_response={
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "probe", "arguments": "{}"},
            }],
        },
        normalized_usage={},
    )
    with pytest.raises(ValueError, match="settled physical attempt"):
        bind_wire_compatibility_receipt(
            candidate=candidate,
            physical_attempt=unresolved,
            semantic_observation=observation,
        )
