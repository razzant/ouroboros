"""Regression tests for the Anthropic empty-text-block / cache_control-on-empty fix.

Anthropic 400s on ``cache_control cannot be set for empty text blocks`` (and rejects
empty tool_result content). An empty tool output sealed as
``{"type":"text","text":"","cache_control":{...}}`` used to make the whole task die as
``provider_unavailable`` with an empty answer. These cover the four touched surfaces:
- ``seal_task_transcript`` (never seal an empty cache anchor)
- ``LLMClient._sanitize_anthropic_tool_result_content`` (empty -> placeholder, keep blocks)
- ``LLMClient._copy_messages_with_cache_policy`` (no cache_control on empty text)
- ``LLMClient._normalize_anthropic_response`` (surface Anthropic stop_reason)
"""

from ouroboros.loop import seal_task_transcript
from ouroboros.llm import LLMClient


# ── seal_task_transcript ──────────────────────────────────────────────
def _build(n_tools: int, empty_idx: int) -> list:
    msgs = [{"role": "user", "content": "q"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": f"a{i}"})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}",
                     "content": "" if i == empty_idx else "x" * 50})
    return msgs


def test_seal_empty_tool_output_keeps_exact_empty_bytes_and_uses_an_earlier_anchor():
    # keep_active=2, 4 tools -> prefer tool 1, but it is empty, so tool 0 seals.
    msgs = _build(4, empty_idx=1)
    seal_task_transcript(msgs, keep_active=2, min_prefix_tokens=0)
    sealed = [m for m in msgs if m.get("role") == "tool" and isinstance(m.get("content"), list)]
    assert len(sealed) == 1, "exactly one cache boundary expected"
    blk = sealed[0]["content"][0]
    assert blk["cache_control"] == {"type": "ephemeral"}
    assert blk["text"].strip(), "sealed cache anchor must never be an empty text block"
    assert blk["text"] == "x" * 50
    assert next(m for m in msgs if m.get("tool_call_id") == "t1")["content"] == ""


def test_seal_nonempty_tool_output_preserved():
    msgs = _build(4, empty_idx=99)  # none empty
    seal_task_transcript(msgs, keep_active=2, min_prefix_tokens=0)
    sealed = [m for m in msgs if m.get("role") == "tool" and isinstance(m.get("content"), list)]
    assert len(sealed) == 1
    assert sealed[0]["content"][0]["text"] == "x" * 50  # real content untouched


# ── _sanitize_anthropic_tool_result_content ───────────────────────────
def test_sanitize_empty_scalar_and_list_become_placeholder():
    S = LLMClient._sanitize_anthropic_tool_result_content
    assert S("") == "(no tool output)"
    assert S("   ") == "(no tool output)"
    assert S(None) == "(no tool output)"
    assert S([]) == "(no tool output)"
    assert S([{"type": "text", "text": ""}]) == "(no tool output)"
    assert S([{"type": "text", "text": "  "}]) == "(no tool output)"


def test_sanitize_keeps_real_content_and_nontext_blocks():
    S = LLMClient._sanitize_anthropic_tool_result_content
    assert S("hello") == "hello"
    # empty text dropped, real text kept
    assert S([{"type": "text", "text": "x"}, {"type": "text", "text": ""}]) == [{"type": "text", "text": "x"}]
    # image / non-text blocks preserved even with no text
    img = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
    assert S(img) == img


# ── _copy_messages_with_cache_policy ──────────────────────────────────
def test_cache_policy_drops_cache_control_on_empty_text_block():
    msg = {"role": "tool", "content": [
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "real", "cache_control": {"type": "ephemeral"}},
    ]}
    out = LLMClient._copy_messages_with_cache_policy(
        [msg], allow_message_cache_control=True, flatten_tool_content_blocks=False
    )[0]["content"]
    empty_blk, real_blk = out[0], out[1]
    assert "cache_control" not in empty_blk, "empty text block must not carry cache_control"
    assert real_blk.get("cache_control") == {"type": "ephemeral"}, "non-empty text keeps cache_control"


def test_cache_policy_keeps_cache_control_on_image_block():
    msg = {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "data": "x"}, "cache_control": {"type": "ephemeral"}},
    ]}
    out = LLMClient._copy_messages_with_cache_policy(
        [msg], allow_message_cache_control=True, flatten_tool_content_blocks=False
    )[0]["content"]
    assert out[0].get("cache_control") == {"type": "ephemeral"}, "image block cache_control preserved"


# ── _normalize_anthropic_response (stop_reason surfaced) ───────────────
def test_normalize_surfaces_stop_reason():
    client = LLMClient.__new__(LLMClient)  # converter helper is pure; skip heavy __init__
    resp = {"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0}}
    message, _usage = client._normalize_anthropic_response(resp, {"resolved_model": "m"})
    assert message.get("stop_reason") == "end_turn"


def test_normalize_empty_response_still_surfaces_stop_reason():
    client = LLMClient.__new__(LLMClient)
    resp = {"content": [], "stop_reason": "max_tokens", "usage": {"input_tokens": 0, "output_tokens": 0}}
    message, _usage = client._normalize_anthropic_response(resp, {"resolved_model": "m"})
    assert message.get("stop_reason") == "max_tokens"


def test_normalize_without_stop_reason_omits_key():
    client = LLMClient.__new__(LLMClient)
    resp = {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 0, "output_tokens": 0}}
    message, _usage = client._normalize_anthropic_response(resp, {"resolved_model": "m"})
    assert "stop_reason" not in message


if __name__ == "__main__":  # allow `python tests/test_anthropic_empty_block_fix.py`
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
