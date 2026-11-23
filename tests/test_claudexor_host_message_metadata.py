"""Host-only message markers (the owner-source observation, review feedback) never enter the strict
model-message envelope."""
import asyncio
from copy import deepcopy
import json

import pytest

from ouroboros.llm_claudexor import _request
from tests.test_llm_claudexor import MODEL, retained, setup as claudexor_setup  # noqa: F401
from tests.test_llm_provider_golden import _CASES, _observe


@pytest.mark.parametrize("marker", ["acceptance_observation", "_acceptance_observation", "review_feedback"])
def test_host_marker_is_removed_without_rewriting_content_tools_or_native_payload(marker):
    target = {"source": "codex", "resolved_model": "exact-model"}
    content = [{"type": "text", "text": "Exact acceptance evidence\r\nПривет"}]
    route = {"source": "codex", "model": "exact-model", "credentialProfileId": "personal",
             "accountFingerprint": "account"}
    messages = [{"role": "user", "content": content, marker: {"revision": "host-only"}},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call-one", "type": "function",
                 "function": {"name": "inspect", "arguments": json.dumps({marker: "user argument"})}}],
                 "nativeContinuation": {"route": route, "format": "opaque-v1", "payload": {marker: "native value"}}},
                {"role": "tool", "tool_call_id": "call-one", "content": "Exact tool result"}]
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object",
              "properties": {marker: {"type": "string"}}}}}]
    original, original_tools = deepcopy(messages), deepcopy(tools)
    payload = _request(target, messages, tools, {})
    # ModelMessage in protocol-3 is strict at this outer object only; native
    # continuation and input/tool-schema JSON retain their own opaque fields.
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls", "nativeContinuation"}
    assert all(set(message) <= allowed for message in payload["messages"])
    assert payload["messages"][0] == {"role": "user", "content": content}
    assert payload["messages"][1:] == messages[1:]
    assert payload["tools"] == tools
    assert messages == original and tools == original_tools


def test_direct_provider_projection_also_excludes_host_acceptance_metadata():
    from ouroboros.llm_messages import _MessageShapingMixin

    messages = [{"role": "user", "content": "Exact evidence", "acceptance_observation": {"revision": "host"},
                 "review_feedback": [{"task_id": "host-task", "run_index": 0, "binding_hash": "critic"}]},
                {"role": "assistant", "content": "Answer", "_acceptance_observation": "private"}]
    original = deepcopy(messages)
    sent = _MessageShapingMixin._copy_messages_with_cache_policy(
        messages, allow_message_cache_control=False, flatten_tool_content_blocks=True)
    assert sent == [{"role": "user", "content": "Exact evidence"}, {"role": "assistant", "content": "Answer"}]
    assert messages == original


_WIRE_CASES = {
    "openrouter.dispatch.happy_path", "openrouter.dispatch.no_proxy_client",
    "openrouter.payload.anthropic_family_cache_ttl", "openrouter.payload.gemini_family_cache_control",
    "openrouter.payload.non_cache_family_flattens_tool_blocks", "openrouter.payload.late_system_notice_demoted",
    "openai.dispatch.happy_path", "openai.payload.reasoning_metadata_stripped",
    "compatible.dispatch.happy_path", "cloudru.dispatch.happy_path", "minimax.dispatch.happy_path",
    "local.dispatch.tool_call_from_text", "anthropic.dispatch.happy_path", "gigachat.dispatch.happy_path",
    "fallback.async.preserves_tools", "fallback.body.parameter_rejection_retry",
}


def test_host_feedback_preserves_exact_provider_wire_and_accounting(subtests):
    """Real provider dispatch/builders, golden transports only: no network."""
    for case in [case for case in _CASES if case["id"] in _WIRE_CASES]:
        with subtests.test(case=case):
            spec = deepcopy(case["spec"])
            args = spec["call"].get("kwargs") or spec["call"]["args"]
            for message in args["messages"]:
                message["review_feedback"] = [{"task_id": "host-task", "run_index": 0, "binding_hash": "critic"}]
            before = deepcopy(spec)
            assert _observe(spec) == case["expected"]
            assert spec == before


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_claudexor_send_copy_preserves_canonical_feedback(claudexor_setup, asynchronous):  # noqa: F811
    from ouroboros.loop_llm_call import _send_main_candidate

    root, gateway, client = claudexor_setup
    messages = [{"role": "user", "content": "Exact reviewer result\r\nПривет", "review_feedback": [
        {"task_id": "host-task", "run_index": 0, "binding_hash": "critic"}]}]
    before = deepcopy(messages)
    if asynchronous:
        asyncio.run(client.chat_async(messages, MODEL))
    else:
        _send_main_candidate(client, {"messages": messages, "model": MODEL}, model=MODEL,
            use_local=False, deadline_ts=None, physical_context=None, candidate_predicate=None)
    payload = gateway.uploads[0][0]
    assert payload["messages"] == [{"role": "user", "content": before[0]["content"]}]
    assert retained(root, "request") == payload
    assert messages == before
    assert len(gateway.creates) == 1 and len(gateway.acks) == 1


def test_deepseek_send_copy_keeps_reasoning_echo_and_user_json(monkeypatch):
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-fixture-key")
    client = LLMClient()
    target = client._resolve_remote_target("deepseek::deepseek-v4-pro")
    arguments = json.dumps({"review_feedback": "user argument"})
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "Inspecting"}],
                 "reasoning_content": "Exact prior reasoning", "tool_calls": [{"id": "call-one", "type": "function",
                 "function": {"name": "inspect", "arguments": arguments}}]},
                {"role": "tool", "tool_call_id": "call-one", "content": [{"type": "text", "text": "Tool result"}]},
                {"role": "user", "content": "Actual reviewer result"}]
    tools = [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object",
        "properties": {"review_feedback": {"type": "string"}}}}}]
    expected = client._build_remote_kwargs(target, messages, "high", 128, "auto", None, tools)
    for message in messages:
        message["review_feedback"] = [{"task_id": "host-task", "run_index": 0, "binding_hash": "critic"}]
    before = deepcopy(messages)
    sent = client._build_remote_kwargs(target, messages, "high", 128, "auto", None, tools)
    assert sent == expected and messages == before
    assert sent["messages"][0]["reasoning_content"] == "Exact prior reasoning"
    assert sent["messages"][0]["tool_calls"][0]["function"]["arguments"] == arguments


def test_main_contributions_stay_out_of_claudexor_wire_even_if_report_fails(claudexor_setup, monkeypatch):  # noqa: F811
    from pathlib import Path
    from types import SimpleNamespace

    from ouroboros import context_assembly_trace, loop_llm_call
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND
    from ouroboros.loop_round_limits import _note_owner_contribution
    from ouroboros.physical_attempt_context import PhysicalAttemptContext
    from tests._usage_store_testing import ledger_rows

    root, gateway, client = claudexor_setup
    gateway.results.append(deepcopy(gateway.results[0]))
    gateway.dispatch.append(gateway.dispatch[0])
    messages = [{'role': 'user', 'content': 'My incoming message', HOST_CONTEXT_KIND_KEY: OWNER_WORDS_KIND}]
    addr = 'row:1@2026-11-23T00:25:00+00:00#1234567890ab'
    slot = SimpleNamespace(last_owner_delivery={'source_row_address': addr})
    _note_owner_contribution(slot, messages[0], 'direct_input')
    monkeypatch.setattr(loop_llm_call, 'current_model_wait', lambda: SimpleNamespace(tool_context=slot))
    context = PhysicalAttemptContext('owner_low', 'low', 'cold_estimate', '', '', None, None, False, False)

    def send():
        loop_llm_call._send_main_candidate(client, {'messages': messages, 'model': MODEL}, model=MODEL,
            use_local=False, deadline_ts=None, physical_context=context, candidate_predicate=None,
            canonical_messages=messages)
        manifests = [row['candidate_manifest_ref'] for row in ledger_rows(root) if row.get('candidate_manifest_ref')]
        return json.loads(Path(manifests[-1]['path']).read_text(encoding='utf-8'))

    facts = send()
    assert facts['late_contributions']['entries'] == [{
        'family': 'owner_input', 'form': 'direct_input', 'path': '$.messages[0]',
        'row_address': addr, 'status': 'carrier_located'}]

    def broken(*args):
        raise RuntimeError('optional report failed')

    monkeypatch.setattr(context_assembly_trace, 'prepared_inventory', broken)
    facts = send()
    assert facts['prepared_inventory'] == {'status': 'unavailable'}
    assert len(gateway.creates) == len(gateway.acks) == 2
    assert all(upload[0]['messages'] == [{'role': 'user', 'content': 'My incoming message'}] for upload in gateway.uploads)
    assert HOST_CONTEXT_KIND_KEY in messages[0]
