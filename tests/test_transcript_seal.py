"""Tests for seal_task_transcript — transcript cache-boundary sealing."""

from ouroboros.loop import seal_task_transcript
from ouroboros.loop_messages import _extract_plain_text_from_content


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _tool_msg(text: str, call_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _user_msg(text: str = "hi") -> dict:
    return {"role": "user", "content": text}


def _assistant_msg(text: str = "thinking") -> dict:
    return {"role": "assistant", "content": text}


def _make_messages(n_tool_rounds: int, prefix_per_tool: int = 3000) -> list:
    """Build a message list with n_tool_rounds tool results of ~prefix_per_tool chars each."""
    msgs = [_user_msg("start" * 100)]
    for i in range(n_tool_rounds):
        msgs.append(_assistant_msg(f"round {i}"))
        msgs.append(_tool_msg("x" * prefix_per_tool, call_id=f"tc{i}"))
    return msgs


def _sealed_indices(messages: list) -> list:
    return [
        i for i, m in enumerate(messages)
        if m.get("role") == "tool" and isinstance(m.get("content"), list)
    ]


# ─────────────────────────────────────────────────────────────
# _extract_plain_text_from_content
# ─────────────────────────────────────────────────────────────

def test_extract_plain_text_string():
    assert _extract_plain_text_from_content("hello") == "hello"


def test_extract_plain_text_list():
    blocks = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
    assert _extract_plain_text_from_content(blocks) == "foobar"


def test_extract_plain_text_none():
    assert _extract_plain_text_from_content(None) == ""


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — basic cases
# ─────────────────────────────────────────────────────────────

def test_no_seal_when_not_enough_tool_rounds():
    """With <= keep_active tool rounds, nothing is sealed."""
    msgs = _make_messages(n_tool_rounds=5, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    assert _sealed_indices(msgs) == []


def test_seal_appears_with_sufficient_rounds_and_prefix():
    """Seal appears when there are more than keep_active tool rounds and prefix is large."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    sealed = _sealed_indices(msgs)
    assert len(sealed) == 1


def test_no_seal_when_prefix_too_short():
    """No seal when prefix tokens are below min_prefix_tokens."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=1)
    # Each tool result is 1 char → prefix will be tiny
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=2048)
    assert _sealed_indices(msgs) == []


def test_sealed_message_has_cache_control():
    """Sealed message has cache_control in its content block."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    idx = _sealed_indices(msgs)[0]
    content = msgs[idx]["content"]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "cache_control" in content[0]
    assert content[0]["cache_control"]["type"] == "ephemeral"


def test_sealed_text_matches_original():
    """Sealed message text is identical to the original tool result.

    With keep_active=5 and 7 total tool rounds, the seal boundary is
    tool_indices[-(5+1)] = tool_indices[-6] = the 2nd tool result (tc1).
    """
    boundary_text = "BOUNDARY_RESULT_" * 200  # the message that will be sealed
    msgs = [
        _user_msg("start" * 200),
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc0"),   # index -7, older than boundary
        _assistant_msg(),
        _tool_msg(boundary_text, "tc1"), # index -6, this is the seal boundary
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc2"),
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc3"),
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc4"),
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc5"),
        _assistant_msg(),
        _tool_msg("x" * 3000, "tc6"),
    ]
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    sealed = _sealed_indices(msgs)
    assert len(sealed) == 1
    assert msgs[sealed[0]]["content"][0]["text"] == boundary_text


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — rotation and revert
# ─────────────────────────────────────────────────────────────

def test_seal_rotates_forward_on_second_call():
    """When new tool rounds are added, the seal boundary advances."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    first_sealed_idx = _sealed_indices(msgs)[0]

    # Add one more tool round
    msgs.append(_assistant_msg("new round"))
    msgs.append(_tool_msg("new_result" * 500, call_id="tc_new"))

    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    second_sealed_idx = _sealed_indices(msgs)[0]
    assert second_sealed_idx > first_sealed_idx


def test_previous_seal_reverted_on_recompute():
    """After a second call, the previously sealed message is plain string again."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    first_sealed_idx = _sealed_indices(msgs)[0]

    # Add a new round so the seal boundary should move
    msgs.append(_assistant_msg("new"))
    msgs.append(_tool_msg("new_data" * 500, call_id="tc_new"))

    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)

    # The previously sealed message must now be a plain string
    assert isinstance(msgs[first_sealed_idx]["content"], str)


def test_only_one_sealed_boundary_at_a_time():
    """There is at most one sealed tool message at any point."""
    msgs = _make_messages(n_tool_rounds=10, prefix_per_tool=3000)
    # Call twice to simulate two consecutive LLM rounds
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)
    assert len(_sealed_indices(msgs)) <= 1


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — compaction safety
# ─────────────────────────────────────────────────────────────

def test_strip_cache_control_flattens_tool_role_list_to_string():
    """Provider cache cleanup flattens tool-role list content to a plain string."""
    from ouroboros.llm import LLMClient
    msgs = [
        {"role": "tool", "tool_call_id": "tc1", "content": [
            {"type": "text", "text": "hello ", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "world"},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "user", "cache_control": {"type": "ephemeral"}},
        ]},
    ]
    cleaned = LLMClient._copy_messages_with_cache_policy(
        msgs,
        allow_message_cache_control=False,
        flatten_tool_content_blocks=True,
    )
    # Tool role should be flattened to plain string
    assert cleaned[0]["content"] == "hello world"
    # Non-tool list content should just lose cache_control
    assert isinstance(cleaned[1]["content"], list)
    assert "cache_control" not in cleaned[1]["content"][0]


def test_anthropic_messages_pass_through_list_content_for_tool_result():
    """_build_anthropic_messages preserves list content for tool_result (Anthropic supports blocks)."""
    from ouroboros.llm import LLMClient
    client = LLMClient()
    sealed_content = [{"type": "text", "text": "sealed data", "cache_control": {"type": "ephemeral"}}]
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "tc1",
            "type": "function",
            "function": {"name": "my_tool", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "tc1", "content": sealed_content},
    ]
    _, anthropic_msgs = client._build_anthropic_messages(messages)
    # The last user block should contain a tool_result with list content
    last_user = next(m for m in reversed(anthropic_msgs) if m["role"] == "user")
    tool_result_block = next(
        b for b in last_user["content"] if b.get("type") == "tool_result"
    )
    assert tool_result_block["content"] == sealed_content


def test_compaction_receives_plain_strings():
    """After seal + revert cycle, all tool messages are plain strings (safe for compaction)."""
    msgs = _make_messages(n_tool_rounds=8, prefix_per_tool=3000)
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)

    # Simulate a new round and re-seal (which reverts previous seal)
    msgs.append(_assistant_msg("another"))
    msgs.append(_tool_msg("more_data" * 300, call_id="tc_extra"))
    seal_task_transcript(msgs, keep_active=5, min_prefix_tokens=100)

    # All tool messages except the current seal boundary should be plain strings
    sealed = set(_sealed_indices(msgs))
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i not in sealed:
            assert isinstance(m["content"], str), f"Tool msg at {i} is not a plain string"
