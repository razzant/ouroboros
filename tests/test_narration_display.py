"""Provider-agnostic narration: LLMClient.extract_display_reasoning reads readable reasoning by
SHAPE and skips opaque/encrypted payloads, so empty tool-round bubbles get narrated without ever
touching the transcript or the round-trip-sensitive metadata."""

from ouroboros.llm import LLMClient


def test_flat_reasoning_string():
    assert LLMClient.extract_display_reasoning({"reasoning": "  thinking about X  "}) == "thinking about X"


def test_reasoning_details_readable_types():
    msg = {"reasoning_details": [
        {"type": "reasoning.text", "text": "step one"},
        {"type": "reasoning.summary", "summary": "summary two"},
    ]}
    assert LLMClient.extract_display_reasoning(msg) == "step one\nsummary two"


def test_reasoning_details_encrypted_is_skipped():
    msg = {"reasoning_details": [
        {"type": "reasoning.encrypted", "data": "BASE64OPAQUE=="},
        {"type": "reasoning.text", "text": "visible"},
    ]}
    # opaque encrypted contributes nothing; only the readable text shows.
    assert LLMClient.extract_display_reasoning(msg) == "visible"


def test_anthropic_thinking_block_read_redacted_skipped():
    msg = {"content": [
        {"type": "thinking", "thinking": "let me reason", "signature": "sig"},
        {"type": "redacted_thinking", "data": "OPAQUE"},
        {"type": "text", "text": "the answer"},
    ]}
    # the readable thinking is surfaced; redacted (opaque) and the plain answer text are not reasoning.
    assert LLMClient.extract_display_reasoning(msg) == "let me reason"


def test_gemini_thought_part():
    msg = {"content": [
        {"thought": True, "text": "gemini thought"},
        {"text": "regular part"},
    ]}
    assert LLMClient.extract_display_reasoning(msg) == "gemini thought"


def test_no_reasoning_returns_empty_and_string_content_is_safe():
    assert LLMClient.extract_display_reasoning({"content": "plain string answer"}) == ""
    assert LLMClient.extract_display_reasoning({}) == ""
    assert LLMClient.extract_display_reasoning(None) == ""


def test_does_not_mutate_message():
    msg = {"reasoning": "x", "content": [{"type": "thinking", "thinking": "y"}]}
    before = dict(msg)
    LLMClient.extract_display_reasoning(msg)
    # display-only: the reader never adds/removes fields (transcript boundary stays clean).
    assert msg == before


def test_visible_round_text_string_and_list_never_reprs():
    from ouroboros.loop_messages import _visible_round_text

    assert _visible_round_text("  hi  ") == "hi"
    # a list of provider blocks joins ONLY text blocks — never a raw Python list repr.
    assert _visible_round_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    # a thinking/thought-only list has NO visible text → reads empty so narration can fall back.
    assert _visible_round_text([{"type": "thinking", "thinking": "x"}, {"thought": True, "text": "y"}]) == ""
    # a regular Gemini part carries `text` with NO `type` — it is still visible answer text, and a
    # sibling thought block is excluded (visible text is the complement of display reasoning).
    assert _visible_round_text([{"thought": True, "text": "pondering"}, {"text": "the answer"}]) == "the answer"
    assert _visible_round_text(None) == ""
    assert _visible_round_text({"type": "text"}) == ""
