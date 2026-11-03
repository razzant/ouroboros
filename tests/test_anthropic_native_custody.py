"""Exact direct-Anthropic custody, replay, and consumer regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest

import ouroboros.config as config
import ouroboros.request_wire_contract as wire
from ouroboros import context_compaction as compaction
from ouroboros.anthropic_native_custody import (
    ANTHROPIC_CONSUMED_RECEIPTS_KEY,
    ANTHROPIC_CUSTODY_PROJECTION_KEY,
    ANTHROPIC_NATIVE_RECEIPT_KEY,
    ANTHROPIC_OPAQUE_PROJECTION_KEY,
    anthropic_replay_scope,
    context_custody_proxy,
    native_content_for_replay,
    public_custody_projection,
    retain_native_assistant_content,
    scrub_native_custody,
)
from ouroboros.context_budget import ContextReclaimRequest
from ouroboros.context_compaction import _atomic_units, _summary_projection
from ouroboros.llm import LLMClient, _canonical_candidate_bytes
from ouroboros.observability import persist_physical_candidate, read_blob_ref
from ouroboros.usage_accounting import (
    UsageScope,
    last_physical_attempt_capture,
    usage_scope,
)


def _target(model="claude-future", endpoint="https://api.anthropic.example/v1"):
    return {
        "provider": "anthropic",
        "resolved_model": model,
        "usage_model": f"anthropic/{model}",
        "base_url": endpoint,
        "contract_headers": {"anthropic-version": "2023-06-01"},
        "api_key": "test-only",
        "default_headers": {},
    }


def _raw_content():
    return [
        {"type": "thinking", "thinking": "opaque thought", "signature": "sig-A"},
        {"type": "text", "text": "visible-before"},
        {"type": "redacted_thinking", "data": "redacted-A"},
        {
            "type": "tool_use",
            "id": "tool-A",
            "name": "probe_a",
            "input": {"value": 1},
            "caller": {"type": "direct", "future": {"opaque": True}},
        },
        {"type": "text", "text": "visible-between", "future_text_field": 7},
        {
            "type": "tool_use",
            "id": "tool-B",
            "name": "probe_b",
            "input": {"value": 2},
            "future_tool_field": ["kept"],
        },
    ]


def _canonical_message(target=None):
    message = {
        "role": "assistant",
        "content": "visible-beforevisible-between",
        "tool_calls": [
            {
                "id": "tool-A",
                "type": "function",
                "function": {"name": "probe_a", "arguments": '{"value":1}'},
            },
            {
                "id": "tool-B",
                "type": "function",
                "function": {"name": "probe_b", "arguments": '{"value":2}'},
            },
        ],
    }
    return retain_native_assistant_content(message, _raw_content(), target or _target())


def _tool_results():
    return [
        {"role": "tool", "tool_call_id": "tool-A", "content": "result-A"},
        {"role": "tool", "tool_call_id": "tool-B", "content": "result-B"},
    ]


def _manifest(ref):
    path = pathlib.Path(ref["path"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    return json.loads(raw)


def _encoded(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False, default=str,
    )


def test_interleaved_raw_blocks_replay_exactly_and_tamper_or_route_drift_scrubs():
    target = _target()
    message = _canonical_message(target)
    receipt = message[ANTHROPIC_NATIVE_RECEIPT_KEY]
    assert json.loads(receipt["content_json"]) == _raw_content()

    with anthropic_replay_scope():
        replay = native_content_for_replay(message, target, ("tool-A", "tool-B"))
        assert replay == _raw_content()
        replay[0]["signature"] = "caller-mutated-copy"
        assert json.loads(receipt["content_json"])[0]["signature"] == "sig-A"

    assert native_content_for_replay(message, target, ("tool-B", "tool-A")) is None
    assert native_content_for_replay(message, _target(model="claude-other"), (
        "tool-A", "tool-B",
    )) is None
    assert native_content_for_replay(message, _target(endpoint="https://other.example/v1"), (
        "tool-A", "tool-B",
    )) is None

    tampered = copy.deepcopy(message)
    tampered[ANTHROPIC_NATIVE_RECEIPT_KEY]["content_json"] = (
        tampered[ANTHROPIC_NATIVE_RECEIPT_KEY]["content_json"].replace("sig-A", "sig-X")
    )
    assert native_content_for_replay(tampered, target, ("tool-A", "tool-B")) is None
    changed_call = copy.deepcopy(message)
    changed_call["tool_calls"][0]["function"]["name"] = "other_tool"
    assert native_content_for_replay(changed_call, target, ("tool-A", "tool-B")) is None
    switched = LLMClient.sanitize_reasoning_on_model_switch(
        [message], "anthropic/claude-future", "anthropic/claude-other"
    )
    assert ANTHROPIC_NATIVE_RECEIPT_KEY not in switched[0]
    assert ANTHROPIC_NATIVE_RECEIPT_KEY not in scrub_native_custody([message])[0]


def test_cache_finalizer_never_mutates_exact_native_replay(monkeypatch):
    target = _target()
    raw = _raw_content()
    raw[1]["cache_control"] = {"type": "ephemeral"}
    message = retain_native_assistant_content(_canonical_message(target), raw, target)
    client = LLMClient(api_key="test")
    monkeypatch.setattr(config, "resolve_prompt_cache_ttl", lambda: "1h")

    with anthropic_replay_scope():
        _system, messages = client._build_anthropic_messages(
            [message, *_tool_results()], target,
        )
        assert messages[0]["content"] == raw
        client._normalize_payload_cache_ttl(target, {"messages": messages})
        assert messages[0]["content"] == raw


def test_builder_places_exact_content_immediately_before_matching_results():
    target = _target()
    message = _canonical_message(target)
    messages = [message, *_tool_results()]
    with anthropic_replay_scope():
        system, physical = LLMClient(api_key="unused")._build_anthropic_messages(
            messages, target
        )
    assert system == []
    assert physical[0] == {"role": "assistant", "content": _raw_content()}
    assert physical[1]["role"] == "user"
    assert [block["tool_use_id"] for block in physical[1]["content"]] == [
        "tool-A", "tool-B",
    ]


def test_complete_tool_unit_stays_active_until_successful_continuation_consumes_it():
    target = _target()
    message = _canonical_message(target)
    messages = [message, *_tool_results()]
    assert _atomic_units(messages) == ()

    client = LLMClient(api_key="unused")
    with anthropic_replay_scope():
        client._build_anthropic_messages(messages, target)
        empty, _ = client._normalize_anthropic_response(
            {"content": [], "stop_reason": "refusal", "usage": {}}, target
        )
    assert ANTHROPIC_CONSUMED_RECEIPTS_KEY not in empty
    assert _atomic_units(messages) == ()

    with anthropic_replay_scope():
        client._build_anthropic_messages(messages, target)
        continuation, _ = client._normalize_anthropic_response({
            "content": [{"type": "text", "text": "settled continuation"}],
            "stop_reason": "end_turn",
            "usage": {},
        }, target)
    assert continuation[ANTHROPIC_CONSUMED_RECEIPTS_KEY] == [
        message[ANTHROPIC_NATIVE_RECEIPT_KEY]["content_sha256"]
    ]
    settled = [*messages, continuation]
    units = _atomic_units(settled)
    assert len(units) == 1
    assert (units[0].start, units[0].end) == (0, 2)


def test_summary_context_and_observability_projections_never_expose_opaque_values():
    message = _canonical_message()
    for projection in (
        _summary_projection(message),
        public_custody_projection(message),
        context_custody_proxy(message),
    ):
        encoded = json.dumps(projection, sort_keys=True)
        assert "opaque thought" not in encoded
        assert "sig-A" not in encoded
        assert "redacted-A" not in encoded
    proxy = context_custody_proxy(message)
    assert ANTHROPIC_NATIVE_RECEIPT_KEY not in proxy
    assert len(proxy["_anthropic_native_opaque_bytes"]) > 100
    assert ANTHROPIC_NATIVE_RECEIPT_KEY in message


def test_real_physical_candidate_cas_projects_native_opaque_fields(
    tmp_path, monkeypatch,
):
    import requests

    monkeypatch.setenv("OUROBOROS_OBSERVABILITY_KEEP_RAW", "1")
    target = _target()
    sent = []

    def post(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return _NativeResponse()

    monkeypatch.setattr(requests, "post", post)
    with usage_scope(UsageScope(drive_root=tmp_path, task_id="native-physical-cas")):
        LLMClient(api_key="unused")._chat_anthropic(
            target,
            [_canonical_message(target), *_tool_results()],
            None,
            "high",
            64,
            "auto",
        )

    capture = last_physical_attempt_capture()
    assert capture is not None and capture.candidate_manifest_ref
    manifest = _manifest(capture.candidate_manifest_ref)
    assert manifest["candidate_raw_sha256"] == hashlib.sha256(
        _canonical_candidate_bytes(sent[0])
    ).hexdigest()
    assert manifest["anthropic_native_custody_projected"] is True
    assert manifest["full_payload_redacted"] is True
    assert manifest["full_payload_custody"] == "redacted_projection_cas"
    assert manifest["full_payload_ref"] == manifest["redacted_projection_ref"]
    projected = read_blob_ref(tmp_path, manifest["full_payload_ref"])
    assert sent[0]["messages"][0]["content"] == _raw_content()
    sent_blocks = sent[0]["messages"][0]["content"]
    blocks = projected["messages"][0]["content"]
    assert [block["type"] for block in blocks] == [
        block["type"] for block in sent_blocks
    ]
    for raw, block in zip(sent_blocks, blocks):
        for metadata in block.get(ANTHROPIC_OPAQUE_PROJECTION_KEY, []):
            value = raw[metadata["field"]]
            encoded = _encoded(value)
            assert metadata["value_type"]
            assert metadata["size_bytes"] == len(encoded.encode("utf-8"))
            assert metadata["sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
    serialized = json.dumps(projected, ensure_ascii=False)
    for opaque in (
        "opaque thought", "sig-A", "redacted-A", "direct", "kept",
    ):
        assert opaque not in serialized


def test_physical_custody_projection_is_native_tool_use_scoped(tmp_path):
    candidate = {
        "model": "generic-model",
        "messages": [{
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "ordinary-visible-detail"}],
        }],
    }
    persisted = persist_physical_candidate(
        tmp_path,
        task_id="generic-cas",
        attempt_id="generic-attempt",
        candidate=candidate,
        candidate_facts={},
    )
    manifest = _manifest(persisted["manifest_ref"])
    assert manifest["anthropic_native_custody_projected"] is False
    assert read_blob_ref(tmp_path, manifest["full_payload_ref"]) == candidate


def test_real_checkpoint_cas_keeps_private_raw_and_projects_receipt_metadata(tmp_path):
    native = _canonical_message()
    receipt = native[ANTHROPIC_NATIVE_RECEIPT_KEY]
    continuation = {
        "role": "assistant",
        "content": "settled",
        ANTHROPIC_CONSUMED_RECEIPTS_KEY: [receipt["content_sha256"]],
    }
    messages = [native, *_tool_results(), continuation]
    request = ContextReclaimRequest(
        route_fp="native-route",
        round_id="round-1",
        transcript_sha256=compaction.context_reclaim_transcript_sha256(messages),
        measurement_basis="cold_estimate",
        measurement_density=1.0,
        reclaim_goal_tokens=1,
        allow_partial_shrink=True,
    )
    selection = compaction._Selection(compaction._atomic_units(messages), "n" * 64, 0)
    assert selection.units
    checkpoint_ref = compaction._persist_reclaim_checkpoint(
        messages,
        request,
        selection,
        drive_root=tmp_path,
        task_id="native-checkpoint-cas",
    )
    assert checkpoint_ref is not None
    assert checkpoint_ref["root"] == "artifact_store"
    from ouroboros.artifacts import read_actor_source_bytes

    actor_payload = json.loads(read_actor_source_bytes(
        tmp_path, "native-checkpoint-cas", checkpoint_ref,
    ))
    assert actor_payload["messages"] == messages
    manifests = sorted(
        (tmp_path / "observability" / "calls" / "native-checkpoint-cas").glob("*.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["full_payload_redacted"] is False
    assert manifest["full_payload_custody"] == "private_unredacted_cas"
    assert manifest["full_payload_ref"] != manifest["redacted_projection_ref"]
    full = read_blob_ref(tmp_path, manifest["full_payload_ref"])
    projection = read_blob_ref(tmp_path, manifest["redacted_projection_ref"])
    assert full["messages"] == messages
    assert full["messages"][0][ANTHROPIC_NATIVE_RECEIPT_KEY] == receipt
    projected_native = projection["messages"][0][ANTHROPIC_CUSTODY_PROJECTION_KEY]
    assert projected_native == {
        "block_types": [block["type"] for block in _raw_content()],
        "size_bytes": len(receipt["content_json"].encode("utf-8")),
        "sha256": receipt["content_sha256"],
    }
    serialized = json.dumps(projection, ensure_ascii=False)
    assert ANTHROPIC_NATIVE_RECEIPT_KEY not in serialized
    assert ANTHROPIC_CONSUMED_RECEIPTS_KEY not in serialized
    assert "opaque thought" not in serialized
    assert "sig-A" not in serialized
    assert "redacted-A" not in serialized


class _RejectDisabled(RuntimeError):
    status_code = 400
    body = {"error": {
        "message": "thinking.type: Input should be 'adaptive'",
        "type": "invalid_request",
    }}


class _NativeResponse:
    def __init__(self, *, reject=False, invalid_json=False):
        self.reject = reject
        self.invalid_json = invalid_json
        self.status_code = 400 if reject else 200
        self.text = "thinking.type: Input should be 'adaptive'" if reject else ""
        self.reason = "Bad Request" if reject else "OK"
        self.url = "https://api.anthropic.example/v1/messages"

    def json(self):
        if self.reject:
            return copy.deepcopy(_RejectDisabled.body)
        if self.invalid_json:
            raise ValueError("invalid json")
        return {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {},
        }


def test_requested_none_sends_disabled_then_success_confirmed_provider_default(
    tmp_path, monkeypatch,
):
    root = tmp_path / "success"
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: root)
    sent = []

    def post(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return _NativeResponse(reject=len(sent) == 1)

    import requests

    monkeypatch.setattr(requests, "post", post)
    client = LLMClient(api_key="unused")
    with usage_scope(UsageScope(drive_root=tmp_path / "usage-success", task_id="native")):
        message, usage = client._chat_anthropic(
            _target(), [{"role": "user", "content": "hi"}], None,
            "none", 64, "auto",
        )
    assert message["content"] == "ok"
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in sent[1]
    assert all("budget_tokens" not in json.dumps(payload) for payload in sent)
    assert usage["request_wire"]["requested_effort"] == "none"
    assert usage["request_wire"]["applied_effort"] == "provider_default"
    state = json.loads((root / "state" / wire.REQUEST_WIRE_STATE_FILE).read_text())
    actions = [record["action"] for entry in state["profiles"].values()
               for record in entry["records"]]
    assert actions == [{
        "fields": ["thinking"],
        "kind": "drop_field",
        "reason_code": "provider_unsupported_field",
    }]


def test_invalid_json_after_repair_commits_no_anthropic_evidence(tmp_path, monkeypatch):
    root = tmp_path / "invalid"
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: root)
    sent = []

    def post(*_args, **kwargs):
        sent.append(copy.deepcopy(kwargs["json"]))
        return _NativeResponse(reject=len(sent) == 1, invalid_json=len(sent) == 2)

    import requests

    monkeypatch.setattr(requests, "post", post)
    with usage_scope(UsageScope(drive_root=tmp_path / "usage-invalid", task_id="native")):
        with pytest.raises(ValueError, match="invalid json"):
            LLMClient(api_key="unused")._chat_anthropic(
                _target(), [{"role": "user", "content": "hi"}], None,
                "none", 64, "auto",
            )
    assert len(sent) == 2
    assert not (root / "state" / wire.REQUEST_WIRE_STATE_FILE).exists()


def test_main_preserves_private_receipt_but_persists_only_public_projection(
    tmp_path, monkeypatch,
):
    import ouroboros.loop_llm_call as main_driver

    message = _canonical_message()
    persisted = []

    class FakeLlm:
        def chat(self, **_kwargs):
            return copy.deepcopy(message), {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost": 0.0,
                "cost_final": True,
                "provider": "anthropic",
                "resolved_model": "anthropic/claude-future",
            }

    def persist(*_args, **kwargs):
        persisted.append(copy.deepcopy(kwargs["payload"]))
        return {}

    monkeypatch.setattr(main_driver, "persist_call", persist)
    monkeypatch.setattr(main_driver, "emit_llm_usage_event", lambda *_a, **_k: None)
    monkeypatch.setattr(main_driver, "_prepare_main_messages", lambda messages, **_k: messages)
    logs = tmp_path / "main" / "logs"
    logs.mkdir(parents=True)
    result, _ = main_driver.call_llm_with_retry(
        FakeLlm(),
        [message, *_tool_results()],
        "anthropic/claude-future",
        [],
        "high",
        1,
        logs,
        "task",
        1,
        None,
        {},
    )
    assert ANTHROPIC_NATIVE_RECEIPT_KEY in result
    assert len(persisted) == 2
    assert all("opaque thought" not in json.dumps(item) for item in persisted)
    assert all(ANTHROPIC_NATIVE_RECEIPT_KEY not in json.dumps(item) for item in persisted)
