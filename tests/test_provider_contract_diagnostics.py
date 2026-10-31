"""Offline evidence proofs through physical persistence and the real SSE assembler."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_stream import consume_stream
from tests.provider_contract_ci import (
    ProviderCanary, delegate_start_canary_arguments, full_registry_canary_tools,
    run_provider_contract_canary,
)
from tests.provider_contract_diagnostics import ProviderEvidence, provider_evidence

CANARY = ProviderCanary("synthetic", "fixture-model", "openrouter", "OPENROUTER_API_KEY", True, "none")
NONCE = "offline-nonce"


def _call(arguments):
    return {"id": "call-canary", "type": "function",
            "function": {"name": "delegate_start", "arguments": json.dumps(arguments)}}


def _wire(arguments, *, partial=False, private="opaque-provider-native-value"):
    raw = json.dumps(arguments)
    pieces = [raw[:len(raw)//2], raw[len(raw)//2:]]
    chunks = [{"id": "response-fixture", "choices": [{"index": 0, "delta": {
        "role": "assistant", "tool_calls": [{"index": 0, "id": "call-canary", "type": "function",
        "function": {"name": "delegate_start", "arguments": piece}}] if index == 0 else [
        {"index": 0, "function": {"arguments": piece}}],
        "reasoning_details": [{"type": "reasoning.encrypted", "data": private, "signature": private}],
    }, "finish_reason": None}]} for index, piece in enumerate(pieces)]
    if not partial:
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                       "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.01}})
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks) + (
        b"" if partial else b"data: [DONE]\n\n")


class Wire:
    headers = {}
    def __init__(self, body):
        self.body = body
    def iter_bytes(self):
        for offset in range(0, len(self.body), 4096):
            yield self.body[offset:offset + 4096]
    def close(self):
        pass


class PhysicalClient:
    """Use the normal attempt/seal and assembler without a vendor/network client."""
    def __init__(self, responses, *, transform=None):
        self.responses, self.transform, self.sends = iter(responses), transform, 0
        self.client = LLMClient(api_key="synthetic-never-sent")

    def chat(self, **kwargs):
        payload = {key: value for key, value in kwargs.items() if key not in {"no_proxy", "timeout", "bypass_response_cache"}}
        target = {"provider": "openrouter", "usage_model": CANARY.model, "resolved_model": CANARY.model,
                  "base_url": "https://provider.invalid/v1", "supports_generation_cost": False,
                  "supports_openrouter_extensions": True}
        def send(**candidate):
            self.sends += 1
            value = next(self.responses)
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, bytes):
                return Wire(value)
            if candidate.get("stream"):
                chunk = {"choices": [{"index": index, "delta": choice["message"],
                                      "finish_reason": choice["finish_reason"]}
                                     for index, choice in enumerate(value["choices"])],
                         "usage": value.get("usage", {})}
                return Wire(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
            return SimpleNamespace(model_dump=lambda: copy.deepcopy(value))
        with ua.capture_attempt_ids() as ids:
            body = self.client._create_chat_completion_with_retries(send, payload, target)
            message, usage = self.client._normalize_remote_response(body.model_dump(), target, skip_cost_fetch=True)
        usage["ledger_attempt_ids"] = list(ids)
        if self.transform:
            message = self.transform(message)
        return message, usage


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: 0.01)
    monkeypatch.setattr("ouroboros.pricing.estimate_cost_optional", lambda *a, **k: 0.01)
    monkeypatch.setattr("tests.provider_contract_ci.time.sleep", lambda *_: None)
    token = ua._LAST_PHYSICAL_ATTEMPT.set(None)
    recorder = ProviderEvidence(directory=tmp_path / "public", root=tmp_path / "private",
        task_id="provider-evidence-fixture", nodeid="test_provider[fixture]", canary=CANARY,
        secrets=("arbitrary-private-credential",))
    try:
        yield recorder
    finally:
        ua._LAST_PHYSICAL_ATTEMPT.reset(token)


def _run(evidence, client, *, canary=CANARY):
    error = None
    with ua.usage_scope(ua.UsageScope(drive_root=evidence.root, task_id=evidence.task_id)):
        try:
            run_provider_contract_canary(client, canary=canary, tools=full_registry_canary_tools(),
                                         nonce=NONCE, observer=evidence)
        except BaseException as exc:
            error = exc
    path = evidence.finish(outcome="failed" if error else "passed", error=error)
    return json.loads(path.read_text(encoding="utf-8")), error


def _physical(record, logical=0, physical=0):
    return record["attempts"][logical]["physical_attempts"][physical]


def _summary(evidence, record):
    from tests.ci_evidence import render_summary, write_json
    outcome = record["outcome"]
    write_json(evidence.directory / "results.json", {"reports": [{
        "nodeid": evidence.nodeid, "phase": "call", "outcome": outcome,
        "canary_id": evidence.canary.canary_id}], "session_exit_code": 1 if outcome == "failed" else 0})
    return render_summary(evidence.directory,
        producer_outcome="failure" if outcome == "failed" else "success",
        artifact_outcome="success", artifact_url="https://github.com/example/artifact")


@pytest.mark.parametrize("change,violation", [
    (lambda value: {"prompt": value["prompt"]}, "arguments_exact_keys"),
    (lambda value: {**value, "retry_of": "", "max_seconds": 0}, "arguments_exact_keys"),
    (lambda value: {**value, "prompt": value["prompt"] + "!"}, "arguments_prompt"),
])
def test_failure_localizes_original_assembled_and_normalized_arguments(evidence, change, violation):
    expected = delegate_start_canary_arguments(NONCE)
    actual = change(expected)
    client = PhysicalClient([_wire(actual)])
    record, error = _run(evidence, client)
    assert isinstance(error, AssertionError) and client.sends == 1
    assert record["error"]["provider_contract_violation"]["violation"] == violation
    assert record["logical_expected"]["value"]["arguments"] == expected
    physical = _physical(record)
    assert physical["physical_request"]["status"] == "available"
    fragments = physical["received_tool_fields"]
    received = "".join(row["fields"].get("arguments", "") for row in fragments["value"])
    assert json.loads(received) == actual
    partial = physical["partial_assembly"]["value"]["choices"][0]["message"]
    normalized = record["attempts"][0]["normalized_response"]["value"]
    assert partial["tool_calls"][0]["function"]["arguments"] == normalized["tool_calls"][0]["function"]["arguments"]
    public = json.dumps(record)
    assert "opaque-provider-native-value" not in public and "wire_base64" not in public
    assert "private_wire_ref" not in public and str(evidence.root) not in public


def test_deliberately_changed_normalized_reply_remains_distinguishable(evidence):
    expected = delegate_start_canary_arguments(NONCE)
    def change(message):
        message["tool_calls"][0]["function"]["arguments"] = json.dumps({"prompt": "changed after wire"})
        return message
    record, error = _run(evidence, PhysicalClient([_wire(expected)], transform=change))
    assert error
    received = "".join(row["fields"].get("arguments", "") for row in _physical(record)["received_tool_fields"]["value"])
    assert json.loads(received) == expected
    assert "changed after wire" in record["attempts"][0]["normalized_response"]["value"]["tool_calls"][0]["function"]["arguments"]


def test_success_compact_and_first_semantic_empty_retry_retained(evidence):
    empty = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.01}}
    client = PhysicalClient([empty, _wire(delegate_start_canary_arguments(NONCE))])
    record, error = _run(evidence, client)
    assert error is None and client.sends == 2 and record["outcome"] == "passed"
    first, second = record["attempts"]
    assert first["status"] == "semantic_empty" and first["detail_retained"]
    assert _physical(record)["received_tool_fields"]["value"] == []
    assert second["status"] == "response_received" and not second["detail_retained"]
    assert "normalized_response" not in second
    assert first["attempt_ids"] != second["attempt_ids"]


def test_ordinary_success_retains_no_payloads(evidence):
    record, error = _run(evidence, PhysicalClient([_wire(delegate_start_canary_arguments(NONCE))]))
    assert error is None and "logical_expected" not in record
    assert not record["attempts"][0]["detail_retained"]
    assert "physical_attempts" not in record["attempts"][0]
    assert record["diagnostics_errors"] == []
    assert "diagnostics_incomplete" not in _summary(evidence, record)


def test_continuation_keeps_separate_request_and_nonstream_gap(evidence):
    bad_final = {"choices": [{"message": {"role": "assistant", "content": "wrong final"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.01}}
    client = PhysicalClient([_wire(delegate_start_canary_arguments(NONCE)), bad_final])
    record, error = _run(evidence, client, canary=replace(CANARY, continue_to_final=True))
    assert error and client.sends == 2
    assert [item["turn"] for item in record["attempts"]] == ["tool", "continuation"]
    assert record["error"]["provider_contract_violation"]["violation"] == "continuation_marker"
    assert _physical(record, 1)["received_tool_fields"]["reason"] == "original_body_unavailable"
    assert record["logical_expected"]["value"]["final_marker"] == f"FULL_REGISTRY_CONTINUED_{NONCE}"
    assert record["diagnostics_errors"] == []
    assert "diagnostics_incomplete" not in _summary(evidence, record)


def test_partial_stream_retains_assembly_without_unsafe_fragments_or_resend(evidence):
    client = PhysicalClient([_wire({"prompt": "partial"}, partial=True)])
    record, error = _run(evidence, client)
    assert error and client.sends == 1
    physical = _physical(record)
    assert physical["stream_complete"] is False
    assert physical["partial_assembly"]["status"] == "available"
    assert physical["received_tool_fields"]["reason"] == "incomplete_stream_fragment_view_omitted"
    assert record["attempts"][0]["error"]["classification"] == "inconclusive"


@pytest.mark.parametrize("secret", ["arbitrary-private-credential", "sk-proj-token-shaped-secret-secret"])
def test_secret_across_fragments_is_not_reconstructible(evidence, secret):
    evidence._secrets = (*evidence._secrets, secret)
    actual = {"prompt": secret, "subagent_id": "provider-contract-canary"}
    raw = _wire(actual)
    # Split inside the secret itself, independent of socket fragmentation.
    prefix = json.dumps(actual).split(secret)[0]
    a, b = secret[:len(secret)//2], secret[len(secret)//2:]
    chunks = [prefix + a, b + json.dumps(actual).split(secret)[1]]
    frames = []
    for index, args in enumerate(chunks):
        frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
            **({"id": "call-secret", "type": "function"} if index == 0 else {}),
            "function": {**({"name": "delegate_start"} if index == 0 else {}), "arguments": args}}]},
            "finish_reason": None}]})
    frames.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    raw = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames) + b"data: [DONE]\n\n"
    record, error = _run(evidence, PhysicalClient([raw]))
    assert error
    encoded = json.dumps(record)
    assert secret not in encoded and a not in encoded and b not in encoded
    assert _physical(record)["received_tool_fields"]["reason"] == "tool_fields_require_redaction; fragments_omitted"
    assert "***" in encoded and "provider-contract-canary" in encoded
    assert record["diagnostics_errors"] == []
    assert "diagnostics_incomplete" not in _summary(evidence, record)


def test_large_safe_payload_complete_and_projection_digest_matches(evidence):
    arguments = {"prompt": "large-safe-content-" * 2000, "subagent_id": "provider-contract-canary"}
    record, error = _run(evidence, PhysicalClient([_wire(arguments)]))
    assert error
    view = record["attempts"][0]["normalized_response"]
    serialized = json.dumps(view["value"], ensure_ascii=False, sort_keys=True, indent=2).encode()
    assert view["projection_bytes"] == len(serialized)
    assert view["projection_sha256"] == hashlib.sha256(serialized).hexdigest()
    assert json.loads(view["value"]["tool_calls"][0]["function"]["arguments"]) == arguments


def test_no_attempt_capture_is_explicit(evidence):
    class NoCapture:
        def chat(self, **_kwargs):
            return {"role": "assistant", "tool_calls": [_call({"prompt": "mismatch"})]}, {}
    record, error = _run(evidence, NoCapture())
    assert error and record["attempts"][0]["physical_evidence"]["reason"] == "no_physical_attempt_identity"


def test_http_error_and_chain_text_never_exported(evidence):
    error = RuntimeError("arbitrary-private-credential")
    error.status_code = 400
    error.body = {"message": "arbitrary-private-credential"}
    error.__cause__ = ValueError("arbitrary-private-credential")
    client = PhysicalClient([error])
    record, returned = _run(evidence, client)
    assert returned is error and client.sends == 1
    assert record["error"]["classification"] == "red" and record["error"]["status_code"] == 400
    assert "arbitrary-private-credential" not in json.dumps(record)


def test_native_private_blocks_never_exported(evidence):
    from pathlib import Path
    raw = (Path(__file__).parent / "fixtures/llm_wire/anthropic/claude-sonnet-5/native_stream.sse").read_bytes()
    from ouroboros.observability import read_call_payload
    capture = ua.PhysicalAttemptCapture("native-attempt", "fixture", "anthropic", "dispatched", "canonical_json_v1")
    with ua.usage_scope(ua.UsageScope(drive_root=evidence.root, task_id=evidence.task_id)):
        ua.adopt_physical_attempt_capture(capture)
        wire = Wire(raw)
        wire.iter_content = lambda **_kwargs: wire.iter_bytes()
        consume_stream(wire, native=True)
    _, payload, _ = read_call_payload(evidence.root, task_id=evidence.task_id, call_id="physical_native-attempt_stream")
    from ouroboros.observability import read_blob_ref
    source = read_blob_ref(evidence.root, payload["private_wire_ref"])
    projected = evidence.physical("native-attempt", {})
    public = json.dumps(projected)
    for block in source["partial_assembly"]["content"]:
        if block["type"] == "thinking":
            assert block["signature"] not in public
            if block["thinking"]:
                assert block["thinking"] not in public
    assert projected["received_tool_fields"]["status"] == "available"
    assert "oura" in public and "boros" in public


def test_export_failure_preserves_original_exception_and_send_count(evidence, monkeypatch):
    from tests import ci_evidence
    config = SimpleNamespace(getoption=lambda key, default=None: str(evidence.directory))
    request = SimpleNamespace(config=config, node=SimpleNamespace(nodeid=evidence.nodeid))
    monkeypatch.setattr(ci_evidence, "output_dir", lambda _config: evidence.directory)
    monkeypatch.setattr(ci_evidence, "write_json", lambda *_args: (_ for _ in ()).throw(OSError("secret-error")))
    original = AssertionError("original canary violation")
    with pytest.raises(AssertionError) as caught:
        with provider_evidence(request, CANARY):
            raise original
    assert caught.value is original


def test_missing_credential_lifecycle_records_skip_without_dispatch(evidence, monkeypatch):
    from tests import ci_evidence
    from tests.provider_contract_ci import require_provider_canary_credential
    monkeypatch.setattr(ci_evidence, "output_dir", lambda _config: evidence.directory)
    monkeypatch.delenv(CANARY.credential_env, raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    request = SimpleNamespace(config=None, node=SimpleNamespace(nodeid=evidence.nodeid))
    with pytest.raises(pytest.skip.Exception):
        with provider_evidence(request, CANARY):
            require_provider_canary_credential(CANARY)
    record = json.loads(next(evidence.directory.glob("provider-*.json")).read_text(encoding="utf-8"))
    assert record["outcome"] == "skipped" and record["attempts"] == []
    assert record["physical_evidence"]["reason"] == "canary_not_dispatched"


@pytest.mark.parametrize("stage", ["physical_request", "stream_projection"])
def test_corrupt_recorded_blob_is_a_gap_and_does_not_replace_red(evidence, monkeypatch, stage):
    from ouroboros import observability
    client = PhysicalClient([_wire({"prompt": "wrong"})])
    initial, error = _run(evidence, client)
    attempt_id = initial["attempts"][0]["attempt_ids"][0]
    if stage == "physical_request":
        manifest, _payload, _ = observability.read_call_payload(
            evidence.root, task_id=evidence.task_id, call_id=attempt_id)
        target_ref = manifest["redacted_projection_ref"]
    else:
        _manifest, payload, _ = observability.read_call_payload(
            evidence.root, task_id=evidence.task_id, call_id=f"physical_{attempt_id}_stream")
        target_ref = payload["private_wire_ref"]
    original = observability.read_blob_ref
    def broken(root, ref, **kwargs):
        if ref["sha256"] == target_ref["sha256"]:
            raise ValueError("private-corruption-message")
        return original(root, ref, **kwargs)
    monkeypatch.setattr(observability, "read_blob_ref", broken)
    evidence.directory = evidence.directory / "corrupt"
    path = evidence.finish(outcome="failed", error=error)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(error, AssertionError) and client.sends == 1
    view = "physical_request" if stage == "physical_request" else "received_tool_fields"
    assert _physical(record)[view]["status"] == "unavailable"
    assert record["diagnostics_errors"] == [{"stage": stage, "attempt_id": attempt_id, "error_type": "ValueError"}]
    summary = _summary(evidence, record)
    assert "diagnostics_incomplete" in summary and "Producer step: **failure**" in summary
    assert "private-corruption-message" not in json.dumps(record) + summary


def test_observer_failure_changes_neither_send_count_nor_original_assertion(evidence):
    class BrokenObserver:
        def observe(self, *_args, **_kwargs):
            raise OSError("private-observer-error")
    client = PhysicalClient([_wire({"prompt": "wrong"})])
    with ua.usage_scope(ua.UsageScope(drive_root=evidence.root, task_id=evidence.task_id)):
        with pytest.raises(AssertionError) as caught:
            run_provider_contract_canary(client, canary=CANARY, tools=full_registry_canary_tools(),
                                         nonce=NONCE, observer=BrokenObserver())
    assert caught.value.args[0]["provider_contract_violation"]["violation"] == "arguments_exact_keys"
    assert client.sends == 1


def test_nested_and_malformed_secret_echoes_are_redacted_with_public_args_preserved(evidence):
    value = {"nested": json.dumps({"items": [{"prompt": "arbitrary-private-credential",
        "subagent_id": "public-actor", "api_key": "not-a-known-format"}]}),
        "malformed": '{"prompt": "arbitrary-private-credential',
        "opaque": "sk-proj-1234567890abcdefghijklmnop"}
    public, changed = evidence.safe(value)
    text = json.dumps(public)
    assert changed and "arbitrary-private-credential" not in text
    assert "not-a-known-format" not in text and "sk-proj-1234567890abcdefghijklmnop" not in text
    assert "public-actor" in text


def test_multiple_physical_ids_in_single_call_do_not_hide_earlier_attempt(evidence):
    class TwoPhysicalCalls:
        def __init__(self):
            self.inner = PhysicalClient([_wire({"prompt": "first"}), _wire(delegate_start_canary_arguments(NONCE))])
        def chat(self, **kwargs):
            _first, first_usage = self.inner.chat(**kwargs)
            second, second_usage = self.inner.chat(**kwargs)
            second_usage["ledger_attempt_ids"] = first_usage["ledger_attempt_ids"] + second_usage["ledger_attempt_ids"]
            return second, second_usage
    client = TwoPhysicalCalls()
    record, error = _run(evidence, client)
    assert error is None and client.inner.sends == 2
    assert len(record["attempts"]) == 1 and len(record["attempts"][0]["physical_attempts"]) == 2
    assert record["attempts"][0]["detail_retained"]
    first_args = _physical(record)["partial_assembly"]["value"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(first_args) == {"prompt": "first"}


@pytest.mark.parametrize("status,reason,outcome", [
    (429, "rate_limit_429", "skipped"),
    (503, "provider_5xx", "skipped"),
    (400, "provider_contract_or_unclassified", "failed"),
])
def test_integration_skip_context_and_summary_preserve_original_classification(
    evidence, monkeypatch, status, reason, outcome,
):
    import httpx
    from tests import test_provider_integration
    error = httpx.HTTPStatusError("synthetic unavailable", request=httpx.Request("POST", "https://provider.invalid"),
                                  response=httpx.Response(status, text="synthetic unavailable"))
    calls = []
    class Client:
        def chat(self, **kwargs):
            calls.append(kwargs)
            raise error
    monkeypatch.setenv(CANARY.credential_env, "public-test-key")
    monkeypatch.setattr(test_provider_integration, "_get_llm_client", Client)
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(ci_evidence_dir=str(evidence.directory))),
                              node=SimpleNamespace(nodeid=evidence.nodeid))
    expected_exception = pytest.skip.Exception if outcome == "skipped" else httpx.HTTPStatusError
    with pytest.raises(expected_exception) as caught:
        test_provider_integration.test_full_registry_provider_contract(CANARY, full_registry_canary_tools(), request,
                                                                   monkeypatch)
    if outcome == "failed":
        assert caught.value is error
    record = json.loads(next(evidence.directory.glob("provider-*.json")).read_text(encoding="utf-8"))
    assert len(calls) == 1 and record["outcome"] == outcome
    assert record["classification"] == reason and record["error"]["status_code"] == status
    assert record["attempts"][0]["error"]["reason"] == reason
    if outcome == "skipped":
        assert record["error"]["classification"] == "inconclusive"
        assert record["error"]["exception_type"] == "Skipped"
        assert record["error"]["originating_exception_type"] == "HTTPStatusError"
    else:
        assert record["error"]["classification"] == "red"
    summary = _summary(evidence, record)
    assert f"| {CANARY.canary_id} | {outcome} | {reason} |" in summary


def test_exhausted_semantic_empty_retry_keeps_specific_summary_cause(evidence):
    from tests.provider_contract_ci import CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS
    empty = {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 20, "completion_tokens": 10, "cost": 0.01}}
    client = PhysicalClient([empty] * CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS)
    record, error = _run(evidence, client)
    assert isinstance(error, AssertionError) and client.sends == CANARY_EMPTY_RESPONSE_MAX_ATTEMPTS
    assert record["outcome"] == "failed" and record["violation"] == "semantic_empty_provider_response"
    assert all(attempt["status"] == "semantic_empty" for attempt in record["attempts"])
    assert f"| {CANARY.canary_id} | failed | semantic_empty_provider_response |" in _summary(evidence, record)


def test_actual_integration_success_remains_compact_and_unclassified(evidence, monkeypatch):
    from tests import test_provider_integration
    client = PhysicalClient([_wire(delegate_start_canary_arguments(NONCE))])
    monkeypatch.setenv(CANARY.credential_env, "public-test-key")
    monkeypatch.setattr(test_provider_integration, "_get_llm_client", lambda: client)
    monkeypatch.setattr(test_provider_integration, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=NONCE)))
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(ci_evidence_dir=str(evidence.directory))),
                              node=SimpleNamespace(nodeid=evidence.nodeid))
    test_provider_integration.test_full_registry_provider_contract(CANARY, full_registry_canary_tools(), request,
                                                                   monkeypatch)
    record = json.loads(next(evidence.directory.glob("provider-*.json")).read_text(encoding="utf-8"))
    assert client.sends == 1 and record["outcome"] == "passed"
    assert record["error"] is None and "classification" not in record
    assert record["diagnostics_errors"] == [] and not record["attempts"][0]["detail_retained"]
    assert "logical_expected" not in record
    assert "diagnostics_incomplete" not in _summary(evidence, record)
