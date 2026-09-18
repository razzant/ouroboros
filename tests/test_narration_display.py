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


def test_final_text_response_redacts_before_delivery_and_trace():
    from ouroboros.loop import _handle_text_response

    candidate = "sk-" + "B2" * 20
    content = f"I removed {candidate} from the payload."
    trace = {"reasoning_notes": []}

    delivered, _, updated = _handle_text_response(content, trace, {})

    assert candidate not in delivered
    assert candidate not in updated["reasoning_notes"][0]
    assert "***REDACTED***" in delivered


def test_round_and_final_prose_redact_all_observability_secret_classes():
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
        assert candidate not in delivered
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


def test_skill_review_projection_redacts_reviewer_secret_prose():
    from ouroboros.skill_review import SkillReviewOutcome, render_skill_review_block

    candidate = "sk-" + "C3" * 20
    outcome = SkillReviewOutcome(
        skill_name="redacted-review",
        status="blockers",
        content_hash="a" * 64,
        reviewer_models=["fake/reviewer"],
        findings=[{
            "item": "secret_handling",
            "verdict": "FAIL",
            "severity": "critical",
            "reason": f"Remove {candidate} from the payload.",
            "model": "fake/reviewer",
        }],
    )

    markdown = render_skill_review_block(outcome)

    assert candidate not in markdown
    assert "***REDACTED***" in markdown


def _recording_emitter(calls):
    def emit(text, **kwargs):
        calls.append((text, kwargs))
    return emit


def test_round_progress_emits_reasoning_stamped_before_visible_text(monkeypatch):
    from ouroboros.loop import _emit_round_progress

    monkeypatch.delenv("OUROBOROS_REASONING_SUMMARY", raising=False)
    calls = []
    trace = {"reasoning_notes": []}
    msg = {"reasoning": " weigh the two options ", "content": "the answer"}

    _emit_round_progress("the answer", msg, _recording_emitter(calls), trace)

    # think -> say: reasoning goes out first, stamped, even though visible text exists.
    assert calls == [
        ("weigh the two options", {"meta": {"reasoning": True}, "narration": False}),
        ("the answer", {"narration": True}),
    ]
    # reasoning is display-only: only the visible text reaches the trace.
    assert trace["reasoning_notes"] == ["the answer"]


def test_round_progress_reasoning_off_emits_only_visible_text(monkeypatch):
    from ouroboros.loop import _emit_round_progress

    monkeypatch.setenv("OUROBOROS_REASONING_SUMMARY", "off")
    calls = []
    msg = {"reasoning": "weigh the two options", "content": "the answer"}

    _emit_round_progress("the answer", msg, _recording_emitter(calls), {"reasoning_notes": []})

    assert calls == [("the answer", {"narration": True})]


def test_history_replays_the_reasoning_stamp_on_a_stored_progress_row(tmp_path):
    """A `progress.jsonl` row stamped `reasoning: true` comes back from the history
    projection with the stamp (it is whitelisted in `_PROGRESS_META_FIELDS`); an
    unstamped row carries no `reasoning` key at all."""
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").touch()
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"ts": "2026-09-11T00:00:00Z", "task_id": "t1", "content": "💬 weigh the options", "reasoning": True},
        {"ts": "2026-09-11T00:00:01Z", "task_id": "t1", "content": "💬 the answer"},
    )))
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    thinking, plain = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]
    assert thinking["text"] == "💬 weigh the options" and thinking["reasoning"] is True
    assert plain["text"] == "💬 the answer" and "reasoning" not in plain


def test_reasoning_rows_stay_in_the_file_but_never_reach_the_recent_progress_digest(tmp_path):
    """`## Recent progress` (ouroboros/context.py, `summarize_progress(limit=50)`) is a
    fixed prompt window fed by progress.jsonl. Reasoning is display-only: the stamped row
    stays on disk for UI replay, yet only narration and visible text reach the digest."""
    import json

    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        {"ts": "2026-09-11T00:00:00Z", "task_id": "t1", "text": "💬 weigh the options", "reasoning": True},
        {"ts": "2026-09-11T00:05:00Z", "task_id": "t1", "text": "💬 the answer"},
    )))
    memory = Memory(tmp_path)
    rows = memory.read_jsonl_tail("progress.jsonl", 200)
    assert [r.get("reasoning") for r in rows] == [True, None]  # both rows survive the read
    assert memory.summarize_progress(rows, limit=50) == "⚙️ 00:05 💬 the answer"
