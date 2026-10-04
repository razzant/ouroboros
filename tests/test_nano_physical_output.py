"""The chosen Nano output allowance binds the final native candidate and ledger."""
import json

import pytest

from ouroboros import config, llm_attempt, usage_accounting as ua
from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS as HEADROOM, OWNER_NANO_TARGET_TOKENS as NANO
from ouroboros.request_wire_recovery import current_wire_candidate
from tests.test_processing_transport import transport as _transport

transport = _transport


# The allowance follows the owner's Nano target (SSOT, pinned in test_context_budget_ssot): room under it, then the
# exact headroom boundary; one token more is refused below.
@pytest.mark.parametrize("input_tokens,cap", [(60826, NANO - 60826), (NANO - HEADROOM, HEADROOM)])
def test_nano_actual_custom_tools_are_measured_before_output_seal(transport, monkeypatch, input_tokens, cap):
    _root, client, sent = transport
    monkeypatch.setattr(config, "get_context_mode", lambda: "nano")
    measured = []

    def measure(target, payload):
        assert payload["tools"][0]["type"] in {"function", "custom"}
        assert "inspect" in json.dumps(payload["tools"])
        measured.append(payload)
        return {"input_tokens": input_tokens, "input_is_exact": True,
                "tokenizer_template_provenance": {"source": "exact_test_template"},
                "route_capacity_tokens": 131072, "route_capacity_confirmed": True}

    monkeypatch.setattr(llm_attempt, "_prepared_input_measurement", measure)
    message, _usage = client.chat([{"role": "user", "content": "complete source"}], "openai::test-model",
        max_tokens=65536, processing_preference="standard", context_mode="nano", tools=[{"type": "function", "function": {
            "name": "inspect", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}])
    assert message["content"] == "ok" and measured and len(sent) == 1
    assert sent[0]["max_completion_tokens"] == cap
def test_nano_unknown_template_keeps_route_usable_and_discloses_unproven_bound(transport, monkeypatch):
    root, client, sent = transport
    monkeypatch.setattr(config, "get_context_mode", lambda: "nano")
    message, _usage = client.chat([{"role": "user", "content": "hello"}], "openai::test-model", max_tokens=512, context_mode="nano")
    assert message["content"] == "ok" and sent[0]["max_completion_tokens"] == 512
    assert sent[0]["max_completion_tokens"] == 512


def test_exact_nano_insufficient_headroom_returns_preparation_facts_without_sending(transport, monkeypatch):
    _, client, sent = transport
    monkeypatch.setattr(config, "get_context_mode", lambda: "nano")
    monkeypatch.setattr(llm_attempt, "_prepared_input_measurement", lambda *_: {
        "input_tokens": NANO - HEADROOM + 1, "input_is_exact": True,
        "tokenizer_template_provenance": {"source": "test"},
        "route_capacity_tokens": 131072, "route_capacity_confirmed": True})
    with pytest.raises(ua.PhysicalAttemptPreparationFailed) as failure:
        client.chat([{"role": "user", "content": "actual input"}], "openai::test-model", context_mode="nano")
    assert failure.value.call_context_fit["fit_status"] == "insufficient_headroom"
    assert sent == []


def test_output_rebind_preserves_the_same_canonical_tool_catalog(transport, monkeypatch):
    from ouroboros.request_wire_recovery import request_wire_call_scope

    _, client, _ = transport
    monkeypatch.setattr(config, "get_context_mode", lambda: "nano")
    monkeypatch.setattr(llm_attempt, "_prepared_input_measurement", lambda *_: {
        "input_tokens": 60826, "input_is_exact": True,
        "tokenizer_template_provenance": {"source": "test"},
        "route_capacity_tokens": 131072, "route_capacity_confirmed": True})
    target = client._resolve_remote_target("openai::test-model")
    payload = client._build_remote_kwargs(target, [{"role": "user", "content": "all input"}],
        "high", 65536, "auto", None, [{"type": "function", "function": {
            "name": "inspect", "parameters": {"type": "object"}}}])
    with request_wire_call_scope():
        first = llm_attempt._finalized_physical_candidate(target, payload, "chat.completions")
        catalog = current_wire_candidate().custom_catalog_sha256
        second = llm_attempt._finalized_physical_candidate(target, first, "chat.completions")
        assert first == second and current_wire_candidate().custom_catalog_sha256 == catalog


def test_local_serving_window_uses_live_arguments_not_training_or_fallback(monkeypatch):
    from types import SimpleNamespace
    from ouroboros.local_model import LocalModelManager

    monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
    manager = LocalModelManager.__new__(LocalModelManager)
    manager._proc = None
    manager._status = "offline"
    manager._context_length = 4096
    manager._serving_context_length = 0
    assert manager.serving_context_evidence()["context_window"] is None
    manager._proc = SimpleNamespace(poll=lambda: None, pid=123)
    manager._status = "ready"
    manager._serving_context_length = 16384
    manager._context_length = 131072  # Training metadata returned by /models.
    assert manager.serving_context_evidence() == {
        "context_window": 16384, "confirmed": True, "source": "owned_server_arguments", "process_id": 123}
    manager._proc = SimpleNamespace(poll=lambda: 0, pid=123)
    assert manager.serving_context_evidence()["context_window"] is None


def _physical(profile):
    return ua.PhysicalAttemptContext(
        profile=profile, rendered_mode="nano", measurement_basis="cold_estimate", route_fp="r", round_id="x:round:1",
        target_total_tokens=NANO if profile == "owner_nano" else None, capacity_total_tokens=200_000,
        context_target_miss=False, automatic_pass_used=False)


@pytest.mark.parametrize("profile,input_tokens,cap", [
    # A Nano the window chose: the window minus the input bounds the reply, never the owner's Nano target.
    ("task_local_nano", 146_000, 200_000 - 146_000),
    ("task_local_nano", 80_000, 65_536),  # room in the window: the caller's whole ceiling, not NANO - input
    ("owner_nano", 60_826, NANO - 60_826),  # the owner's Nano keeps its target
])
def test_a_nano_the_window_chose_sizes_its_reply_by_the_window(transport, monkeypatch, profile, input_tokens, cap):
    _root, client, sent = transport
    monkeypatch.setattr(llm_attempt, "_prepared_input_measurement", lambda *_: {
        "input_tokens": input_tokens, "input_is_exact": True, "tokenizer_template_provenance": {"source": "test"},
        "route_capacity_tokens": 200_000, "route_capacity_confirmed": True})
    with ua.bind_physical_attempt_context(_physical(profile)):
        message, _usage = client.chat([{"role": "user", "content": "input"}], "openai::test-model",
                                      max_tokens=65_536, context_mode="nano")
    assert message["content"] == "ok" and sent[0]["max_completion_tokens"] == cap
    assert sent[0]["max_completion_tokens"] + input_tokens <= 200_000
