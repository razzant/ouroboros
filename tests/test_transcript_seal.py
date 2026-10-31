"""Tests for seal_task_transcript — the one message-side cache boundary."""

import copy

from ouroboros.loop import seal_task_transcript
from ouroboros.loop_messages import _extract_plain_text_from_content


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _tool_msg(text, call_id: str = "tc1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _user_msg(text="hi") -> dict:
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


def _marked(messages: list) -> list:
    """Rows carrying the seal's host-only mark, a message-level ``cache_control``."""
    return [i for i, m in enumerate(messages) if "cache_control" in m]


def _unmarked(messages: list) -> list:
    return [{key: value for key, value in m.items() if key != "cache_control"} for m in messages]


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


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
# seal_task_transcript — the newest row with text (#596, #1186)
# ─────────────────────────────────────────────────────────────

def test_the_newest_tool_result_is_the_boundary_without_a_lag():
    """No tool result waits outside the cached prefix, whatever the round count."""
    for rounds in (1, 3, 5, 6, 7):
        msgs = _make_messages(n_tool_rounds=rounds)
        seal_task_transcript(msgs, min_prefix_tokens=100)
        assert _marked(msgs) == [len(msgs) - 1]


def test_the_mark_is_host_only_and_every_row_keeps_its_exact_content():
    msgs = _make_messages(n_tool_rounds=7)
    before = copy.deepcopy(msgs)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert msgs[-1]["cache_control"] == {"type": "ephemeral"}
    assert _unmarked(msgs) == before


def test_tool_free_rows_advance_the_boundary():
    """A clock line and owner words are the newest rows with text; an answer never is."""
    msgs = _make_messages(n_tool_rounds=2)
    msgs += [_user_msg("[Host clock at this request: 2026-10-21T00:00:00Z UTC]"), _assistant_msg("done")]
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [len(msgs) - 2]
    msgs.append(_user_msg("owner: one more thing"))
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [len(msgs) - 1]


def test_empty_opaque_and_textless_rows_fall_back_to_an_earlier_text_row():
    msgs = _make_messages(n_tool_rounds=2)
    anchor = len(msgs) - 1
    msgs += [_assistant_msg(), _tool_msg("", "empty"), _tool_msg("   ", "blank"),
             _tool_msg([{"type": "text", "text": "shot"}, _IMAGE], "opaque"),
             _user_msg([_IMAGE]), _user_msg("   "), _assistant_msg("answer")]
    before = copy.deepcopy(msgs)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [anchor]
    assert _unmarked(msgs) == before  # an empty output stays exact: padding belongs to the send copy


def test_a_multipart_user_row_with_text_is_the_boundary_and_keeps_its_blocks():
    msgs = _make_messages(n_tool_rounds=2) + [_user_msg([{"type": "text", "text": "look"}, _IMAGE])]
    before = copy.deepcopy(msgs)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [len(msgs) - 1] and _unmarked(msgs) == before


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — rotation and revert
# ─────────────────────────────────────────────────────────────

def test_seal_rotates_forward_on_second_call():
    """When new tool rounds are added, the seal boundary advances."""
    msgs = _make_messages(n_tool_rounds=7)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    first_sealed_idx = _marked(msgs)[0]

    msgs.append(_assistant_msg("new round"))
    msgs.append(_tool_msg("new_result" * 500, call_id="tc_new"))

    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [len(msgs) - 1] and len(msgs) - 1 > first_sealed_idx


def test_previous_seal_reverted_on_recompute():
    """After a second call, the previous boundary row is exactly as it was produced."""
    msgs = _make_messages(n_tool_rounds=7)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    first_sealed_idx = _marked(msgs)[0]
    produced = _unmarked([msgs[first_sealed_idx]])[0]

    msgs.append(_assistant_msg("new"))
    msgs.append(_tool_msg("new_data" * 500, call_id="tc_new"))
    seal_task_transcript(msgs, min_prefix_tokens=100)

    assert msgs[first_sealed_idx] == produced


def test_only_one_sealed_boundary_at_a_time():
    """There is exactly one marked row at any point."""
    msgs = _make_messages(n_tool_rounds=10)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert len(_marked(msgs)) == 1


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — send copies and compaction safety
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


def test_marker_routes_render_the_mark_only_in_their_send_copy():
    """A marker route's copy draws the block marker where the old in-place seal put it
    (a tool result's one text block, nested in ``tool_result.content`` on the direct
    lane); a marker-free copy drops the mark; the transcript itself never changes."""
    from ouroboros.llm import LLMClient
    msgs = [_user_msg("start"), {"role": "assistant", "content": None, "tool_calls": [{
        "id": "tc1", "type": "function", "function": {"name": "my_tool", "arguments": "{}"}}]},
        _tool_msg("sealed data")]
    seal_task_transcript(msgs, min_prefix_tokens=0)
    before = copy.deepcopy(msgs)
    marked = [{"type": "text", "text": "sealed data", "cache_control": {"type": "ephemeral"}}]
    copied = LLMClient._copy_messages_with_cache_policy(
        msgs, allow_message_cache_control=True, flatten_tool_content_blocks=False)
    plain = LLMClient._copy_messages_with_cache_policy(
        msgs, allow_message_cache_control=False, flatten_tool_content_blocks=True)
    _, direct = LLMClient()._build_anthropic_messages(msgs)
    assert copied[2] == {"role": "tool", "tool_call_id": "tc1", "content": marked}
    assert plain[2] == {"role": "tool", "tool_call_id": "tc1", "content": "sealed data"}
    assert direct[-1]["content"] == [{"type": "tool_result", "tool_use_id": "tc1", "content": marked}]
    assert msgs == before


def test_claudexor_send_copy_is_stable_when_rolling_seal_moves():
    """The subscription projection drops the mark: each request extends the last one."""
    import json

    from ouroboros.llm_claudexor import _request

    target = {"source": "codex", "resolved_model": "model"}
    parameters = {"model_account_override": ""}
    messages = _make_messages(n_tool_rounds=7, prefix_per_tool=3000)
    seal_task_transcript(messages, min_prefix_tokens=100)
    first = _request(target, messages, None, parameters)["messages"]

    messages.append(_assistant_msg("new round"))
    messages.append(_tool_msg("new_result" * 500, call_id="tc_new"))
    seal_task_transcript(messages, min_prefix_tokens=100)
    second = _request(target, messages, None, parameters)["messages"]

    assert second[:len(first)] == first
    assert "cache_control" not in json.dumps(second)


def test_claudexor_send_copy_preserves_nontext_tool_blocks():
    from ouroboros.llm_claudexor import _request

    content = [
        {"type": "text", "text": "caption"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    payload = _request(
        {"source": "codex", "resolved_model": "model"},
        [{"role": "tool", "tool_call_id": "tc-image", "content": content}],
        None,
        {"model_account_override": ""},
    )
    assert payload["messages"][0]["content"] == content


def test_compaction_receives_plain_strings():
    """Across seals every tool message stays a plain string, the boundary row included."""
    msgs = _make_messages(n_tool_rounds=8, prefix_per_tool=3000)
    seal_task_transcript(msgs, min_prefix_tokens=100)

    msgs.append(_assistant_msg("another"))
    msgs.append(_tool_msg("more_data" * 300, call_id="tc_extra"))
    seal_task_transcript(msgs, min_prefix_tokens=100)

    assert all(isinstance(m["content"], str) for m in msgs if m.get("role") == "tool")


# ─────────────────────────────────────────────────────────────
# seal_task_transcript — the task-message anchor and older transcripts
# ─────────────────────────────────────────────────────────────

def test_a_lone_task_message_is_the_anchor_and_keeps_its_shape():
    msgs = _make_messages(n_tool_rounds=0)
    seal_task_transcript(msgs, min_prefix_tokens=100)
    assert _marked(msgs) == [0] and msgs[0]["content"] == "start" * 100


def test_qualifying_rolling_seal_migrates_the_marker_in_the_same_call():
    """Exactly one message-side marker exists across the migration."""
    msgs = _make_messages(n_tool_rounds=3, prefix_per_tool=4)
    seal_task_transcript(msgs, min_prefix_tokens=1000)
    assert _marked(msgs) == [0]

    msgs.append(_assistant_msg("round 3"))
    msgs.append(_tool_msg("x" * 4000, call_id="tc3"))
    seal_task_transcript(msgs, min_prefix_tokens=1000)

    assert _marked(msgs) == [len(msgs) - 1]


def test_an_unqualified_prefix_keeps_the_task_message_anchor():
    """A newer row with text but a prefix too small keeps one marker on the task."""
    msgs = _make_messages(n_tool_rounds=7, prefix_per_tool=4)
    seal_task_transcript(msgs, min_prefix_tokens=100_000)

    assert _marked(msgs) == [0]


def test_blank_task_message_is_never_the_cache_anchor():
    """Anthropic rejects a marker on empty text; leave it unmarked instead."""
    msgs = [_user_msg("   "), _assistant_msg("a"), _tool_msg("out", call_id="tc0")]
    seal_task_transcript(msgs, min_prefix_tokens=100)

    assert _marked(msgs) == []
    assert msgs[0]["content"] == "   "


def test_the_anchor_lands_on_the_last_non_empty_text_block():
    """A multipart task message keeps its later blocks cacheable too, on every marker route."""
    from ouroboros.llm import LLMClient
    blocks = [{"type": "text", "text": "task"}, {"type": "text", "text": "contract"}, {"type": "text", "text": "   "}]
    msgs = [_user_msg(copy.deepcopy(blocks)), _assistant_msg("a"), _tool_msg("out", call_id="tc0")]
    seal_task_transcript(msgs, min_prefix_tokens=100)

    assert _marked(msgs) == [0] and msgs[0]["content"] == blocks
    copied = LLMClient._copy_messages_with_cache_policy(
        msgs, allow_message_cache_control=True, flatten_tool_content_blocks=False)
    _, direct = LLMClient()._build_anthropic_messages(msgs)
    for content in (copied[0]["content"], direct[0]["content"]):
        assert [block["text"] for block in content if "cache_control" in block] == ["contract"]


def test_an_old_in_place_seal_is_undone_without_another_shape_change():
    """A transcript the old seal wrapped (restored from a checkpoint): the tool row flattens
    to its exact text, the block-list task message keeps its list and its affinity, an opaque
    row stays exact, and the one mark moves to the newest text row."""
    from ouroboros.llm import LLMClient
    marker = {"type": "ephemeral"}
    cited = [{"type": "text", "text": "cited", "citations": [{"url": "https://example.invalid"}]}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": "stable policy", "cache_control": marker}]},
            {"role": "user", "content": [{"type": "text", "text": "task", "cache_control": marker}]},
            _assistant_msg(), _tool_msg([{"type": "text", "text": "old seal", "cache_control": marker}], "t0"),
            _assistant_msg(), _tool_msg(copy.deepcopy(cited), "t1")]
    session = LLMClient._openrouter_session_identity("anthropic/claude-fable-5", msgs)
    seal_task_transcript(msgs, min_prefix_tokens=0)

    assert msgs[1]["content"] == [{"type": "text", "text": "task"}]
    assert msgs[3]["content"] == "old seal" and msgs[5]["content"] == cited
    assert _marked(msgs) == [3]
    assert session and LLMClient._openrouter_session_identity("anthropic/claude-fable-5", msgs) == session


def test_a_caller_declared_marker_on_a_later_row_is_left_in_place():
    from ouroboros.tools.review_helpers import cached_prompt_blocks
    declared = cached_prompt_blocks("stable review contract", "this round's evidence", ttl="1h")
    msgs = [_user_msg("task"), _assistant_msg(), _user_msg(copy.deepcopy(declared)),
            _assistant_msg(), _tool_msg("result")]
    seal_task_transcript(msgs, min_prefix_tokens=0)

    assert msgs[2]["content"] == declared
    assert _marked(msgs) == [4]
