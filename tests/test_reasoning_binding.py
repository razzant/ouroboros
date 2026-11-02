"""Signed reasoning replays only while its answered conversation is unchanged."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros.anthropic_native_custody import (
    ANTHROPIC_CONSUMED_RECEIPTS_KEY,
    ANTHROPIC_NATIVE_RECEIPT_KEY,
    anthropic_replay_scope,
    anthropic_tool_unit_active,
    mark_replayed_receipts_consumed,
    native_content_for_replay,
    retain_native_assistant_content,
)
from ouroboros.reasoning_artifacts import prefix_bound_reasoning, retire_prefix_bound_reasoning
from ouroboros.transcript_prefix import (
    DIGEST_ATTR,
    message_digest,
    observe_send,
    retire_unbound_reasoning,
    tools_digest,
)


SYSTEM = {"role": "system", "content": [{"type": "text", "text": "Complete the owner task."}]}
TASK = {"role": "user", "content": "Read both notes."}
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]


def signed(text="thought", fmt="anthropic-claude-v1"):
    return {"type": "reasoning.text", "text": text, "signature": "sig-" + text,
            "format": fmt, "index": 0}


def turn(call_id, fmt="anthropic-claude-v1"):
    return [{"role": "assistant", "content": "", "reasoning": "thinking " + call_id,
             "reasoning_details": [signed("thinking " + call_id, fmt)],
             "tool_calls": [{"id": call_id, "type": "function", "function": {
                 "name": "read_file", "arguments": json.dumps({"path": call_id}),
             }}]}, {"role": "tool", "tool_call_id": call_id, "content": "body of " + call_id}]


def history():
    return [copy.deepcopy(SYSTEM), copy.deepcopy(TASK), *turn("c1"), *turn("c2"), *turn("c3")]


def answered(messages, tools=TOOLS):
    slot = SimpleNamespace()
    observe_send(slot, messages, round_idx=1, tools=tools)
    return slot


def bound_rows(messages):
    return [index for index, message in enumerate(messages) if prefix_bound_reasoning(message)]


def target(provider="anthropic"):
    return {"provider": provider, "resolved_model": "anthropic/fixture-model" if provider == "openrouter" else "fixture-model",
            "base_url": "https://api.anthropic.example/v1",
            "supports_openrouter_extensions": provider == "openrouter"}


def native_turn(call_id="c1"):
    message, result = turn(call_id)
    blocks = [
        {"type": "thinking", "thinking": "thought", "signature": "native-signature"},
        {"type": "text", "text": "before"},
        {"type": "server_tool_use", "id": "server-" + call_id, "name": "web_search", "input": {"q": "x"}},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "web_search_tool_result", "tool_use_id": "server-" + call_id, "content": []},
        {"type": "tool_use", "id": call_id, "name": "read_file", "input": {"path": call_id}},
        {"type": "text", "text": "after"},
    ]
    message.update(content="beforeafter", reasoning="thought", reasoning_details=[
        signed(), {"type": "reasoning.encrypted", "data": "opaque", "format": "anthropic-claude-v1"},
    ])
    return retain_native_assistant_content(message, blocks, target()), result, blocks


@pytest.mark.parametrize("provider", ["anthropic", "openrouter"])
def test_append_only_rounds_keep_reasoning_through_real_wire_builders(provider):
    from ouroboros.context_fit import seal_task_transcript
    from ouroboros.llm import LLMClient

    client, slot = LLMClient(api_key="test-only"), SimpleNamespace()
    messages = [copy.deepcopy(SYSTEM), copy.deepcopy(TASK)]
    retired = []
    for round_idx in range(1, 5):
        seal_task_transcript(messages, min_prefix_tokens=1)
        fact = retire_unbound_reasoning(slot, messages, TOOLS)
        if fact:
            retired.append(fact)
        before = copy.deepcopy(messages)
        with anthropic_replay_scope():
            payload = client._build_remote_candidate(
                target(provider), messages, "high", 4096, "auto", None, TOOLS,
                skip_capability_fetch=True,
            )
        if provider == "anthropic":
            replayed = sum(block.get("type") in {"thinking", "redacted_thinking"}
                           for row in payload["messages"] for block in row["content"])
            assert replayed == 2 * (round_idx - 1)
        else:
            assert sum(prefix_bound_reasoning(row) for row in payload["messages"]) == round_idx - 1
        assert messages == before
        observe_send(slot, messages, round_idx=round_idx, tools=TOOLS)
        if provider == "anthropic":
            assistant, result, _blocks = native_turn("c" + str(round_idx))
            messages.extend([assistant, result])
        else:
            messages.extend(turn("c" + str(round_idx)))
    assert retired == []
    assert bound_rows(messages) == [2, 4, 6, 8]


@pytest.mark.parametrize("index", [3, 4])
def test_middle_edit_removes_only_reasoning_at_and_after_the_edit(index):
    messages = history()
    slot = answered(messages[:6])
    messages[index]["content"] = "changed visible body"
    earlier = copy.deepcopy(messages[:index])
    visible = [{key: value for key, value in row.items() if key not in {"reasoning", "reasoning_details"}}
               for row in messages]
    assert retire_unbound_reasoning(slot, messages, TOOLS) == {
        "index": index, "cause": "messages", "turns": 2, "blocks": 2,
    }
    assert messages[:index] == earlier
    assert bound_rows(messages) == [2]
    assert [{key: value for key, value in row.items() if key not in {"reasoning", "reasoning_details"}}
            for row in messages] == visible
    assert retire_unbound_reasoning(slot, messages, TOOLS) is None


@pytest.mark.parametrize("cause", ["system", "tools"])
def test_system_or_tool_change_retires_the_whole_bound_run(cause):
    messages, schemas = history(), copy.deepcopy(TOOLS)
    slot = answered(messages[:6])
    if cause == "system":
        messages[0]["content"] = "Updated governance."
    else:
        schemas[0]["function"]["description"] = "Read the current file."
    assert retire_unbound_reasoning(slot, messages, schemas) == {
        "index": 0, "cause": cause, "turns": 3, "blocks": 3,
    }
    assert bound_rows(messages) == []


def test_tools_take_precedence_when_both_system_and_tools_changed():
    messages = history()
    slot = answered(messages[:6])
    messages[0]["content"] = "Updated governance."
    assert retire_unbound_reasoning(slot, messages, []) == {
        "index": 0, "cause": "tools", "turns": 3, "blocks": 3,
    }


def test_the_next_answered_send_becomes_the_baseline_for_new_reasoning():
    messages = history()
    slot = answered(messages[:6])
    messages[3]["content"] = "rewritten result"
    assert retire_unbound_reasoning(slot, messages, TOOLS)["turns"] == 2
    observe_send(slot, messages, round_idx=2, tools=TOOLS)
    messages.extend(turn("c4"))
    assert retire_unbound_reasoning(slot, messages, TOOLS) is None
    assert bound_rows(messages) == [2, 8]


@pytest.mark.parametrize("slot", [None, SimpleNamespace()])
def test_no_baseline_retires_once_without_touching_visible_history(slot):
    messages = history()
    assert retire_unbound_reasoning(slot, messages, TOOLS) == {
        "index": 0, "cause": "no_baseline", "turns": 3, "blocks": 3,
    }
    assert bound_rows(messages) == []
    assert retire_unbound_reasoning(slot, messages, TOOLS) is None


@pytest.mark.parametrize("change", [None, "system", "tools"])
def test_restored_last_answered_observation_supplies_the_baseline(change):
    from ouroboros.owner_wait import continuation_state
    from ouroboros.tools.compact_context import record_context_view

    messages, schemas = history(), copy.deepcopy(TOOLS)
    original = SimpleNamespace(task_id="reasoning-binding-restore")
    record_context_view(original, messages[:6], schemas)
    state = continuation_state(original, messages, {}, {}, 3, schemas, set())
    restored = SimpleNamespace(**state["context_observations"])
    assert not hasattr(restored, DIGEST_ATTR)
    if change == "system":
        messages[0]["content"] = "Freshly rendered governance."
    elif change == "tools":
        schemas[0]["function"]["description"] += " With new semantics."
    fact = retire_unbound_reasoning(restored, messages, schemas)
    if change:
        assert fact == {"index": 0, "cause": change, "turns": 3, "blocks": 3}
        assert bound_rows(messages) == []
    else:
        assert fact is None
        assert bound_rows(messages) == [2, 4, 6]


def test_live_answered_digests_take_precedence_over_an_older_context_observation():
    messages = history()
    slot = answered(messages[:6])
    slot._last_context_observation = {"messages": [{"role": "system", "content": "old"}], "tool_schemas": []}
    assert retire_unbound_reasoning(slot, messages, TOOLS) is None
    assert bound_rows(messages) == [2, 4, 6]


def test_tools_are_a_name_ordered_set_and_only_outer_cache_marker_is_ignored():
    other = {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}}
    schemas = [copy.deepcopy(TOOLS[0]), other]
    reordered = [dict(other, cache_control={"type": "ephemeral"}), copy.deepcopy(TOOLS[0])]
    assert tools_digest(schemas) == tools_digest(reordered)
    messages = history()
    assert retire_unbound_reasoning(answered(messages[:6], schemas), messages, reordered) is None
    for field in ("cache_control", "_private"):
        changed = copy.deepcopy(schemas)
        changed[0]["function"]["parameters"]["properties"][field] = {"type": "string"}
        assert tools_digest(schemas) != tools_digest(changed)
    annotated = copy.deepcopy(schemas)
    annotated[0]["_schema_metadata"] = "schema content"
    assert tools_digest(schemas) != tools_digest(annotated)
    assert tools_digest(schemas) != tools_digest(schemas[:1])


@pytest.mark.parametrize("content", [
    " \nread this\t ",
    [{"type": "text", "text": "read this"}],
    [{"type": "text", "text": ""}, {"type": "text", "text": " read this\n"}, {"type": "text", "text": "  "}],
    [{"type": "input_text", "text": "read this", "cache_control": {"type": "ephemeral"}, "_caption": "private"}],
    [{"type": "output_text", "text": "read this"}],
    ["read this"],
    ["", " read this\n", "  "],
    [{"type": " text ", "text": "read this"}],
])
def test_digest_normalizes_outer_text_whitespace_empty_blocks_and_one_block(content):
    assert message_digest({"role": "user", "content": content}) == message_digest({"role": "user", "content": "read this"})


def test_digest_preserves_text_block_boundaries_and_internal_whitespace():
    merged = {"role": "tool", "tool_call_id": "c1", "content": "ab"}
    split = {**merged, "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert message_digest(merged) != message_digest(split)
    assert message_digest(merged) != message_digest({**merged, "content": "a b"})


def test_digest_ignores_private_carriers_and_reasoning_content_blocks():
    plain = {"role": "assistant", "content": "visible"}
    decorated = {**plain, "_private": {"different": True}, "nativeContinuation": {"state": "opaque"},
                 "reasoning": "flat", "reasoning_details": [signed()], "reasoning_content": "echo",
                 "content": [{"type": "thinking", "thinking": "private", "signature": "signed"},
                             {"type": "text", "text": "visible", "_source_path": "private"},
                             {"type": "redacted_thinking", "data": "opaque"},
                             {"type": "reasoning", "text": "other private reasoning"}]}
    assert message_digest(plain) == message_digest(decorated)


def test_images_are_identified_by_decoded_bytes_or_url():
    encoded = base64.b64encode(b"actual image bytes").decode()
    data_url = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}
    native = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}
    spaced = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded[:8] + "\n" + encoded[8:]}}
    row = lambda block: {"role": "user", "content": [block]}
    assert message_digest(row(data_url)) == message_digest(row(native)) == message_digest(row(spaced))
    changed = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"changed").decode()}}
    assert message_digest(row(data_url)) != message_digest(row(changed))
    for url in ("https://example.test/a.png", "https://example.test/b.png"):
        canonical = row({"type": "image_url", "image_url": {"url": url}})
        assert message_digest(canonical) == message_digest(row({"type": "image", "source": {"type": "url", "url": url}}))
    assert message_digest(row(data_url)) != message_digest(row({"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}))


def test_swapped_image_breaks_the_prefix_and_retires_later_reasoning():
    messages = history()
    messages.insert(2, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}]})
    slot = answered(messages[:7])
    messages[2]["content"][0]["image_url"]["url"] = "https://example.test/b.png"
    assert retire_unbound_reasoning(slot, messages, TOOLS) == {
        "index": 2, "cause": "messages", "turns": 3, "blocks": 3,
    }
    assert bound_rows(messages) == []


@pytest.mark.parametrize("extra", [
    {"reasoning_details": [signed(fmt="google-gemini-v1")]},
    {"reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque", "format": "openai-responses-v1"}]},
    {"reasoning_content": "DeepSeek echo"},
    {"nativeContinuation": {"source": "claudexor", "opaque": "preserved"}},
    {"reasoning": "plain readable reasoning without a bound carrier"},
])
def test_foreign_reasoning_shapes_survive_even_without_a_baseline(extra):
    row = {"role": "assistant", "content": "visible", **extra}
    before = copy.deepcopy(row)
    assert not prefix_bound_reasoning(row)
    assert retire_prefix_bound_reasoning(row) == 0
    assert retire_unbound_reasoning(SimpleNamespace(), [SYSTEM, TASK, row], TOOLS) is None
    assert row == before


def test_mixed_details_keep_foreign_artifacts_and_their_flat_reasoning():
    foreign = signed("foreign", "google-gemini-v1")
    row = {"role": "assistant", "content": "visible", "reasoning": "shared flat text",
           "reasoning_details": [signed(), foreign], "reasoning_content": "echo", "nativeContinuation": {"id": "native"}}
    assert prefix_bound_reasoning(row) == 1
    assert retire_prefix_bound_reasoning(row) == 1
    assert row["reasoning_details"] == [foreign]
    assert row["reasoning"] == "shared flat text"
    assert row["reasoning_content"] == "echo"
    assert row["nativeContinuation"] == {"id": "native"}
    assert retire_prefix_bound_reasoning(row) == 0


@pytest.mark.parametrize("detail", [
    {"type": "reasoning.summary", "summary": "readable", "format": "anthropic-future-v2"},
    {"type": "reasoning.encrypted", "data": "opaque", "format": "anthropic-other-v1"},
])
def test_format_claims_prefix_binding_without_a_signature_or_model_name(detail):
    row = {"role": "assistant", "content": "visible", "reasoning_details": [detail], "reasoning": "derived"}
    assert prefix_bound_reasoning(row) == 1
    assert retire_prefix_bound_reasoning(row) == 1
    assert row == {"role": "assistant", "content": "visible"}


def test_native_content_blocks_retire_without_removing_visible_text():
    row = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": "opaque"},
        {"type": "text", "text": "visible"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]}
    assert prefix_bound_reasoning(row) == 2
    assert retire_prefix_bound_reasoning(row) == 2
    assert row == {"role": "assistant", "content": [{"type": "text", "text": "visible"}]}
    assert retire_prefix_bound_reasoning(row) == 0


def test_native_receipt_is_rehashed_and_replays_server_blocks_in_original_order():
    from ouroboros.llm import LLMClient

    row, result, blocks = native_turn()
    receipt = copy.deepcopy(row[ANTHROPIC_NATIVE_RECEIPT_KEY])
    expected = [block for block in blocks if block["type"] not in {"thinking", "redacted_thinking"}]
    assert prefix_bound_reasoning(row) == 2  # The detail projection does not double-count custody.
    assert retire_prefix_bound_reasoning(row) == 2
    assert "reasoning_details" not in row and "reasoning" not in row
    rewritten = row[ANTHROPIC_NATIVE_RECEIPT_KEY]
    assert {key: rewritten[key] for key in ("version", "route", "tool_use_ids")} == {
        key: receipt[key] for key in ("version", "route", "tool_use_ids")
    }
    assert rewritten["content_sha256"] != receipt["content_sha256"]
    assert rewritten["content_sha256"] == hashlib.sha256(rewritten["content_json"].encode()).hexdigest()
    assert json.loads(rewritten["content_json"]) == expected
    with anthropic_replay_scope():
        assert native_content_for_replay(row, target(), ["c1"]) == expected
        _system, physical = LLMClient(api_key="test-only")._build_anthropic_messages([row, result], target())
    assert physical[0] == {"role": "assistant", "content": expected}
    assert physical[1]["content"][0]["tool_use_id"] == "c1"
    assert not prefix_bound_reasoning(row)
    assert retire_prefix_bound_reasoning(row) == 0
    assert row[ANTHROPIC_NATIVE_RECEIPT_KEY] == rewritten


def test_rewritten_receipt_becomes_active_until_its_new_digest_is_consumed():
    row, result, _blocks = native_turn()
    with anthropic_replay_scope():
        native_content_for_replay(row, target(), ["c1"])
        old_response = mark_replayed_receipts_consumed({"role": "assistant", "content": "old continuation"})
    messages = [row, result, old_response]
    assert not anthropic_tool_unit_active(messages, 0, 1)
    assert retire_prefix_bound_reasoning(row) == 2
    assert anthropic_tool_unit_active(messages, 0, 1)
    with anthropic_replay_scope():
        native_content_for_replay(row, target(), ["c1"])
        new_response = mark_replayed_receipts_consumed({"role": "assistant", "content": "new continuation"})
    assert new_response[ANTHROPIC_CONSUMED_RECEIPTS_KEY] == [row[ANTHROPIC_NATIVE_RECEIPT_KEY]["content_sha256"]]
    assert not anthropic_tool_unit_active([*messages, new_response], 0, 1)
