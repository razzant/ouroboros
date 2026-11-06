"""Production-driver regressions for exact-route request-wire recovery."""

from __future__ import annotations

import copy
import json

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros.request_wire_contract import (
    canonical_sha256,
    infer_tool_dialect,
    physical_candidate_sha256,
)
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    finalize_wire_response,
    note_wire_send_failed,
    note_wire_send_succeeded,
    plan_next_wire_retry,
    plan_wire_retry_from_exception,
    prepare_wire_payload_for_send,
    request_wire_call_scope,
)
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture,
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
