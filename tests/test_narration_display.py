"""Provider-agnostic narration: LLMClient.extract_display_reasoning reads readable reasoning by
SHAPE and skips opaque/encrypted payloads, so empty tool-round bubbles get narrated without ever
touching the transcript or the round-trip-sensitive metadata."""

from ouroboros.llm import LLMClient


def _recorder(sink):
    """A progress sink shaped like the real emitter: ``_emit_round_progress``
    names the round's voice with ``narration=True``, so a bare ``list.append``
    would only prove the fake's signature."""
    def emit(text, **meta):
        sink.append(text)
        emit.meta.append(meta)
    emit.meta = []
    return emit


def test_flat_reasoning_string():
    assert LLMClient.extract_display_reasoning({"reasoning": "  thinking about X  "}) == "thinking about X"


def test_reasoning_details_readable_types():
    msg = {"reasoning_details": [
        {"type": "reasoning.text", "text": "step one"},
        {"type": "reasoning.summary", "summary": "summary two"},
    ]}
    assert LLMClient.extract_display_reasoning(msg) == "step one\nsummary two"


def test_minimax_split_reasoning_narrates_and_leaves_content_parseable():
    """The direct MiniMax lane asks for ``reasoning_split``, so the thinking arrives as
    ``reasoning_details`` and ``content`` is the bare verdict — narration reads the one,
    the review parser reads the other, and no ``<think>`` block has to be stripped."""
    from ouroboros.triad_review import empty_array_is_verified_clean

    msg = {
        "role": "assistant",
        "content": "[]\nNO_FINDINGS",
        "reasoning_details": [{
            "type": "reasoning.text",
            "id": "reasoning-text-1",
            "format": "MiniMax-response-v1",
            "index": 0,
            "text": "checked the diff",
        }],
    }
    assert LLMClient.extract_display_reasoning(msg) == "checked the diff"
    assert empty_array_is_verified_clean(msg["content"]) is True


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
    from ouroboros.loop import _visible_round_text

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


def test_round_progress_redacts_secret_shaped_model_text_before_trace_and_emit():
    from ouroboros.loop import _emit_round_progress

    candidate = "sk-" + "A1" * 20
    visible = f"Removing leaked credential {candidate} before retry."
    progress = []
    trace = {"reasoning_notes": []}

    emit = _recorder(progress)
    _emit_round_progress(visible, {}, emit, trace)

    assert candidate not in progress[0]
    assert candidate not in trace["reasoning_notes"][0]
    assert "***REDACTED***" in progress[0]
    # The round's own text is the turn's voice, so the card may take its title
    # from it; a redaction never demotes the line to a host note.
    assert emit.meta == [{"narration": True}]


def test_final_text_response_is_delivered_as_written_and_only_its_trace_note_is_redacted():
    """The answer is the author's semantic reply (owner, 29.10): delivered whole. Its trace note
    feeds the stored trace summary, a derived projection, and stays redacted."""
    from ouroboros.loop import _handle_text_response

    candidate = "sk-" + "B2" * 20
    content = f"I removed {candidate} from the payload."
    trace = {"reasoning_notes": []}

    delivered, _, updated = _handle_text_response(content, trace, {})

    assert delivered == content
    assert candidate not in updated["reasoning_notes"][0]
    assert "***REDACTED***" in updated["reasoning_notes"][0]


def test_round_progress_and_trace_notes_redact_all_observability_secret_classes():
    from ouroboros.loop import _emit_round_progress, _handle_text_response

    candidates = [
        "ghp_" + "D4" * 20,
        "Bearer " + "E5" * 16,
        "eyJ" + "F6" * 8 + "." + "G7" * 8 + "." + "H8" * 8,
        "https://example.test/?access_token=" + "I9" * 16,
        "api_key = " + "J0" * 16,
    ]
    for candidate in candidates:
        progress = []
        trace = {"reasoning_notes": []}
        content = f"Credential evidence: {candidate}"

        _emit_round_progress(content, {}, _recorder(progress), trace)
        delivered, _, final_trace = _handle_text_response(content, {"reasoning_notes": []}, {})

        assert candidate not in progress[0]
        assert candidate not in trace["reasoning_notes"][0]
        assert delivered == content
        assert candidate not in final_trace["reasoning_notes"][0]

    ordinary = "Edited tool.py and completed a fresh review."
    progress = []
    trace = {"reasoning_notes": []}
    _emit_round_progress(ordinary, {}, _recorder(progress), trace)
    delivered, _, final_trace = _handle_text_response(ordinary, {"reasoning_notes": []}, {})
    assert progress == [ordinary]
    assert trace["reasoning_notes"] == [ordinary]
    assert delivered == ordinary
    assert final_trace["reasoning_notes"] == [ordinary]


def test_skill_review_block_quotes_reviewer_prose_and_the_owner_card_redacts_it(tmp_path):
    """The block the tool returns quotes the reviewer whole; the chat card's detail of a
    history row, an owner projection, stays redacted."""
    import json

    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block
    from ouroboros.skill_review_history import review_history_detail, review_history_path

    candidate = "sk-" + "C3" * 20
    finding = {
        "item": "secret_handling",
        "verdict": "FAIL",
        "severity": "critical",
        "reason": f"Remove {candidate} from the payload.",
        "model": "fake/reviewer",
    }
    outcome = SkillReviewOutcome(
        skill_name="redacted-review",
        status="blockers",
        content_hash="a" * 64,
        reviewer_models=["fake/reviewer"],
        findings=[finding],
    )

    assert f"secret_handling: Remove {candidate} from the payload." in render_skill_review_block(outcome)

    path = review_history_path(tmp_path, "redacted-review")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "job_id": "job-1", "status": "blockers", "content_hash": "a" * 64,
        "raw_actor_records": [{"model_id": "fake/reviewer", "status": "responded", "parsed_items": [finding]}],
    }) + "\n", encoding="utf-8")
    card = review_history_detail(tmp_path, "redacted-review", "job-1")["markdown"]

    assert "secret_handling" in card and candidate not in card
    assert "***REDACTED***" in card
