"""A returned response keeps its request-wire disclosure when settlement fails.

The disclosure states which request shape was physically sent; the attempt
ledger states what it cost. These are separate facts, so a ledger write that
fails after a paid response must not erase the wire disclosure the response
consumers (custom-tool binding, reporting) bind to. Durable compatibility
learning keeps its own strict settled-capture gate.
"""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

import ouroboros.request_wire_contract as wire
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.openai_chat_custom import normalize_openai_custom_tool_calls
from ouroboros.openai_chat_dispatch import (
    CUSTOM_RECEIPTS_USAGE_KEY,
    pop_custom_validation_receipts,
)
from ouroboros.request_wire_attempt import (
    WireUsageDisclosure,
    validate_physical_wire_attempt,
)
from ouroboros.request_wire_contract import canonical_sha256, physical_candidate_sha256
from ouroboros.request_wire_receipts import (
    WireCandidateSpec,
    bind_wire_candidate,
)
from ouroboros.request_wire_recovery import (
    current_wire_candidate,
    finalize_wire_response,
    merge_request_wire_usage,
    note_wire_send_succeeded,
    register_wire_candidate,
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
from tests._usage_store_testing import ledger_rows


class _Rejected(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status_code = status
        self.body = {"error": {"message": message, "type": "invalid_request_error"}}


class _Response:
    def __init__(self, body=None):
        self._body = body or {
            "id": "resp-paid",
            "choices": [{"message": {"role": "assistant", "content": "paid answer"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "cost": 0.002},
        }

    def model_dump(self):
        return copy.deepcopy(self._body)


def _target(provider="openai", model="vendor/future"):
    return {
        "provider": provider,
        "resolved_model": model,
        "usage_model": f"{provider}/{model}",
        "base_url": f"https://{provider}.example/v1",
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
    }


def _payload(*, effort="high"):
    return {
        "model": "vendor/future",
        "messages": [{"role": "user", "content": "probe"}],
        "reasoning_effort": effort,
        "temperature": 0.4,
        "max_tokens": 32,
        "tools": [{
            "type": "function",
            "function": {
                "name": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                },
            },
        }],
        "tool_choice": "auto",
    }


@pytest.fixture
def wire_env(tmp_path, monkeypatch):
    """Isolated data root, evidence store and budget for one real send lane."""
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    evidence = tmp_path / "evidence"
    (evidence / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: evidence)
    capture_token = ua._LAST_PHYSICAL_ATTEMPT.set(None)
    try:
        yield root, evidence
    finally:
        ua._LAST_PHYSICAL_ATTEMPT.reset(capture_token)


def _store(evidence):
    path = evidence / "state" / wire.REQUEST_WIRE_STATE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _learned_actions(evidence):
    data = _store(evidence) or {}
    return [
        record.get("action")
        for entry in (data.get("profiles") or {}).values()
        for record in (entry.get("records") or [])
    ]


def _rows(root):
    return ledger_rows(root)


def _capture(candidate, *, state="settled", attempt="attempt-wire", digest=None,
             provider=None, model=None, manifest=True, measurement="canonical_json_v1"):
    return PhysicalAttemptCapture(
        attempt_id=attempt,
        model=model or candidate.physical_model,
        provider=provider or candidate.accepted_profile.provider,
        state=state,
        candidate_measurement_kind=measurement,
        candidate_raw_sha256=digest or candidate.candidate_sha256,
        candidate_manifest_ref={
            "path": "/private/candidate.json",
            "call_id": attempt,
            "sha256": canonical_sha256("manifest"),
        } if manifest else None,
    )


def _custom_candidate(target):
    payload = _payload()
    return bind_wire_candidate(
        target=target,
        api_surface="chat.completions",
        source_payload=payload,
        candidate_spec=WireCandidateSpec(
            "openai_chat_custom", "high", "requested_wire_form",
        ),
        requested_effort="high",
        ladder_ordinal=1,
    )


def _settlement_failure(monkeypatch, active, *, fail_unresolved):
    """Break only the paid response's ledger writes, never the store or transport."""
    real_settle, real_unresolved = ua.settle_attempt, ua.mark_unresolved

    def settle(reservation, usage=None, **kwargs):
        if active["paid"]:
            raise RuntimeError("ledger settlement append refused")
        return real_settle(reservation, usage, **kwargs)

    def unresolved(reservation, reason):
        if active["paid"] and fail_unresolved:
            raise RuntimeError("ledger unresolved append refused")
        return real_unresolved(reservation, reason)

    monkeypatch.setattr(ua, "settle_attempt", settle)
    monkeypatch.setattr(ua, "mark_unresolved", unresolved)


@pytest.mark.parametrize("settlement", ["settled", "unresolved", "dispatched"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_returned_response_discloses_its_exact_wire_candidate(
    wire_env, monkeypatch, settlement, asynchronous,
):
    """The paid candidate is disclosed whatever the ledger managed to record."""
    root, evidence = wire_env
    target = _target(provider="openrouter")
    client = LLMClient(api_key="unused")
    sent = []
    active = {"paid": False}
    response = _Response()

    def create(**candidate):
        sent.append(copy.deepcopy(candidate))
        if len(sent) == 1:
            # One genuine learnable step-down: the ladder's reactive action is
            # what the settled control must persist and the unsettled runs must not.
            raise _Rejected("reasoning_effort value 'high' is not supported")
        active["paid"] = settlement != "settled"
        return response

    async def create_async(**candidate):
        return create(**candidate)

    if settlement != "settled":
        _settlement_failure(
            monkeypatch, active, fail_unresolved=settlement == "dispatched",
        )

    def _observe(returned):
        """Read the attempt capture and the projections in the sending context."""
        capture = ua.last_physical_attempt_capture()
        before = usage_projection(root)
        message, usage = client._normalize_remote_response(
            returned.model_dump(), target, skip_cost_fetch=True,
        )
        return returned, capture, before, message, usage, usage_projection(root)

    async def _run_async():
        return _observe(await client._create_chat_completion_with_retries_async(
            create_async, _payload(), target,
        ))

    scope = UsageScope(drive_root=root, task_id="wire-task", root_task_id="wire-task")
    with request_wire_call_scope(), usage_scope(scope):
        if asynchronous:
            returned, capture, before, message, usage, after = asyncio.run(_run_async())
        else:
            returned, capture, before, message, usage, after = _observe(
                client._create_chat_completion_with_retries(create, _payload(), target),
            )

    # The paid response survives its accounting outcome, and is sent exactly once.
    assert returned is response
    assert message["content"] == "paid answer"
    # The provider's own usage is normalized untouched by the ledger outcome.
    assert usage["cost"] == 0.002 and usage["cost_final"] is True
    assert "cost_estimated" not in usage
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (11, 3)
    assert [item["reasoning_effort"] for item in sent] == ["high", "medium"]
    assert capture.state == ("settled" if settlement == "settled" else settlement)

    disclosure = usage["request_wire"]
    assert disclosure == {
        "original_requested_effort": None,
        "requested_effort_source": "provider_projection",
        "applied_effort_source": "sent_candidate",
        "reported_effort": None,
        "reported_effort_source": None,
        "requested_effort": "high",
        "applied_effort": "medium",
        "requested_tool_dialect": "function",
        "applied_tool_dialect": "function",
        "reason_code": "provider_prescribed_value",
        "source_profile_fingerprint": disclosure["source_profile_fingerprint"],
        "accepted_profile_fingerprint": disclosure["accepted_profile_fingerprint"],
        "attempt_id": capture.attempt_id,
        "candidate_sha256": physical_candidate_sha256(sent[1]),
        "ladder_ordinal": 2,
        "applied_actions": disclosure["applied_actions"],
        "task_local": False,
    }
    assert [item["action"]["to"] for item in disclosure["applied_actions"]] == ["medium"]

    # The monetary record keeps its own honest answer; disclosing changes nothing.
    assert after == before
    paid = [row for row in _rows(root) if row["attempt_id"] == capture.attempt_id]
    assert paid[-1]["state"] == ("settled" if settlement == "settled" else settlement)
    if settlement == "settled":
        assert paid[-1]["cost_usd"] == 0.002 and paid[-1]["cost_final"] is True
        assert after["settled_usd"] == 0.002
        assert _learned_actions(evidence) and all(
            action["to"] == "medium" for action in _learned_actions(evidence)
        )
    else:
        assert all(row.get("cost_usd") is None for row in paid)
        assert after["settled_usd"] == 0.0 and after["cost_final"] is False
        # Disclosed, never taught: learning needs a settled attempt lifecycle,
        # which is a different fact from a final price.
        assert _store(evidence) is None


def test_returned_provider_error_reports_itself_without_learning(wire_env, monkeypatch):
    """A provider body error keeps its typed report and its factual disclosure."""
    root, evidence = wire_env
    target = _target(provider="openrouter")
    target["supports_openrouter_extensions"] = True
    client = LLMClient(api_key="unused")
    body = {
        "id": "resp-error",
        "choices": [],
        "error": {"code": 502, "message": "upstream unavailable"},
        "usage": {},
    }
    scope = UsageScope(drive_root=root, task_id="error-task", root_task_id="error-task")
    with request_wire_call_scope(), usage_scope(scope):
        returned = client._create_chat_completion_with_retries(
            lambda **_candidate: _Response(body), _payload(), target,
        )
        _message, usage = client._normalize_remote_response(
            returned.model_dump(), target, skip_cost_fetch=True,
        )

    assert usage["provider_error"]["message"] == "upstream unavailable"
    assert usage["request_wire"]["attempt_id"]
    assert usage["request_wire"]["applied_effort"] == "high"
    assert _store(evidence) is None


def test_custom_tool_receipts_bind_to_an_unsettled_disclosed_candidate(wire_env):
    """The same-class consumer: custom-call validation needs the disclosure, not money."""
    _root, evidence = wire_env
    target = _target()
    candidate = _custom_candidate(target)
    capture = _capture(candidate, state="unresolved", attempt="attempt-custom")
    calls, receipts = normalize_openai_custom_tool_calls([{
        "id": "call-1",
        "type": "custom",
        "custom": {"name": "probe", "input": '{"marker":"ok"}'},
    }], candidate)
    message = {"role": "assistant", "content": "", "tool_calls": calls}
    usage = {CUSTOM_RECEIPTS_USAGE_KEY: receipts}

    with request_wire_call_scope():
        register_wire_candidate(
            candidate,
            source_payload=candidate.physical_payload(),
            target=target,
        )
        note_wire_send_succeeded(capture)
        finalize_wire_response(message, usage, custom_receipts=receipts)
        disclosures = request_wire_disclosures()

    assert usage["request_wire"]["attempt_id"] == "attempt-custom"
    assert usage["request_wire"]["candidate_sha256"] == candidate.candidate_sha256
    assert pop_custom_validation_receipts(usage, message["tool_calls"]) == receipts
    assert _store(evidence) is None  # An unsettled capture never teaches the route.

    # Nested aggregation keeps one row per (attempt, candidate) identity.
    total = {}
    merge_request_wire_usage(total, {"request_wire": dict(usage["request_wire"])})
    merge_request_wire_usage(total, {"request_wire": dict(usage["request_wire"])})
    assert len(total["request_wire_history"]) == 1
    assert len(disclosures) == 1


def test_disclosure_requires_exact_identity_while_learning_requires_settlement(wire_env):
    """Identity is the disclosure's only gate; the receipt gate keeps settlement."""
    _root, _evidence = wire_env
    target = _target()
    candidate = _custom_candidate(target)

    for state in ("settled", "unresolved", "dispatched", "reserved", "released"):
        disclosure = WireUsageDisclosure.from_candidate(
            candidate, _capture(candidate, state=state),
        )
        assert disclosure.attempt_id == "attempt-wire"
        assert disclosure.candidate_sha256 == candidate.candidate_sha256

    with pytest.raises(ValueError, match="identified physical attempt"):
        WireUsageDisclosure.from_candidate(candidate, _capture(candidate, attempt=""))
    with pytest.raises(ValueError, match="candidate digest"):
        WireUsageDisclosure.from_candidate(
            candidate, _capture(candidate, digest=canonical_sha256("other")),
        )
    with pytest.raises(ValueError, match="another route"):
        WireUsageDisclosure.from_candidate(
            candidate, _capture(candidate, provider="elsewhere"),
        )
    with pytest.raises(ValueError, match="manifest receipt"):
        WireUsageDisclosure.from_candidate(candidate, _capture(candidate, manifest=False))
    with pytest.raises(ValueError, match="inspectable canonical candidate"):
        WireUsageDisclosure.from_candidate(
            candidate, _capture(candidate, measurement="opaque"),
        )
    with pytest.raises(ValueError, match="physical-attempt capture"):
        WireUsageDisclosure.from_candidate(candidate, object())

    # The compatibility-receipt validator still composes identity with settlement.
    validate_physical_wire_attempt(candidate, _capture(candidate))
    with pytest.raises(ValueError, match="settled physical attempt"):
        validate_physical_wire_attempt(candidate, _capture(candidate, state="released"))


def test_a_refused_pre_dispatch_attempt_stages_no_returned_response(wire_env):
    """A released capture reaches no disclosure because nothing was ever returned."""
    root, evidence = wire_env
    target = _target()
    client = LLMClient(api_key="unused")
    sent = []
    usage = {}

    scope = UsageScope(drive_root=root, task_id="refused", root_task_id="refused")
    with request_wire_call_scope(), usage_scope(scope):
        with physical_attempt_limit(0), pytest.raises(PhysicalAttemptLimitExceeded):
            client._create_chat_completion_with_retries(
                lambda **candidate: sent.append(candidate), _payload(), target,
            )
        assert current_wire_candidate() is not None
        assert ua.last_physical_attempt_capture().state == "released"
        finalize_wire_response({"role": "assistant", "content": ""}, usage)

    assert sent == []
    assert "request_wire" not in usage
    assert _store(evidence) is None
