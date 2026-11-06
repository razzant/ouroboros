"""Two smaller cuts of people's words: a Reply's preview says when it is cut (its address reads the rest; the model
also reads how long the whole message is, while the stored preview stays the contract's), and a result review's
question about a changed request lists every new word of my human whole.

Both are checked both ways: a short text stays as it is, a long one is marked; every word arrives, none twice,
and the message my human replied to is never listed as their words.
"""
from __future__ import annotations

from types import SimpleNamespace

from ouroboros.chat_references import reference_from_row

LONG = "Keep the export path for finance; " * 30 + "and ship only after the review."


def test_a_reply_preview_marks_its_cut_and_leaves_a_short_text_alone():
    from ouroboros.dialogue_provenance import relation_text

    long = reference_from_row({"direction": "in", "chat_id": 1, "ts": "2026-11-04T10:00:00Z", "text": LONG})
    assert long["snippet"] == LONG[:159].rstrip() + "…" and len(long["snippet"]) <= 160, "the contract's preview"
    assert long["chars"] == len(LONG) and relation_text({"reply_to": long}).endswith(
        f": {long['snippet']} [{len(LONG)} chars in full]]"), "only the model's reading states the whole size"
    short = reference_from_row({"direction": "in", "chat_id": 1, "ts": "2026-11-04T10:00:00Z", "text": "  yes \n "})
    assert short["snippet"] == "yes" and "chars" not in short and "chars in full" not in relation_text({"reply_to": short})


def test_a_changed_request_question_lists_every_new_word_of_my_human_whole(monkeypatch):
    from ouroboros.loop_messages import _record_owner_directive
    from ouroboros.tools import review_result

    ctx = SimpleNamespace(task_id="t-1", task_metadata={"delegation_role": "root"}, current_task_type="task")
    words = [f"Word {index}: " + LONG + ("\nsecond line" if index == 0 else "") for index in range(7)]
    for index, text in enumerate(words):
        _record_owner_directive(ctx, source="owner_mailbox", content=text, msg_id=f"mail-{index}",
                                reply_to={"address": "row:1@x#abc", "author": "Sasha", "text": "Sasha's spec"}
                                if index == 0 else None)
    monkeypatch.setattr(review_result, "_retained_subject_text", lambda _ctx, _sha: "the earlier request")
    text = review_result._request_changed_text(ctx, {"record_id": "rl-1", "verdict": {"aggregate": "PASS"},
                                                     "subject": {"diff_sha": "d" * 64}})
    for word in words:
        assert word.replace("\n", "\n  ") in text, "each word whole, a later line indented under its item"
    assert "more message(s)" not in text and "Sasha's spec" not in text
